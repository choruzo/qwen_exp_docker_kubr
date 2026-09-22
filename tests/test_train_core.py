from __future__ import annotations

import json
import sys
from types import SimpleNamespace
import hashlib
from pathlib import Path

import pytest

from docker_k8s_finetune.errors import PipelineError
from docker_k8s_finetune.train.callbacks import build_checkpoint_provenance_callback
from docker_k8s_finetune.benchmark.core import judge_provenance
from docker_k8s_finetune.train.core import (
    build_attempt,
    artifact_files_manifest,
    finalize_gguf_export,
    filter_formatted_by_token_length,
    format_chatml,
    latest_checkpoint,
    load_chatml,
    resolve_model_reference,
    resolve_text_tokenizer,
    verify_local_model_provenance,
    verify_export_artifact,
    inspect_local_model,
    validate_training_config,
)
from docker_k8s_finetune.train.runner import _verify_baseline, _warmup_steps
from docker_k8s_finetune.train import core as train_core
from docker_k8s_finetune.train import runner as train_runner


def _record() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Pregunta"},
            {"role": "assistant", "content": "Respuesta"},
        ],
        "meta": {"source": "fixture", "category": "concepto", "license": "MIT", "url": "https://example.test"},
    }


def _config() -> dict:
    return {
        "model": {"loader": "FastVisionModel", "architecture": "qwen3_5_vlm_text_only", "local_path": "model", "name": "remote", "revision": "a" * 40, "expected_sha256": {"model-1.safetensors": hashlib.sha256(b"{}").hexdigest()}, "prefer_local_if_present": True, "max_seq_length": 8192},
        "architecture": {"text_only": True, "finetune_vision_layers": False},
        "lora": {"target_modules": "auto"},
        "trainer": {"response_only_loss": True, "train_sampling_strategy": "random", "overlength_action": "exclude", "length_audit_batch_size": 256, "per_device_train_batch_size": 1, "per_device_eval_batch_size": 4, "gradient_accumulation_steps": 16, "effective_batch_size": 16, "eval_steps": 500, "oom_fallback": {"enabled": True, "profiles": [{"name": "context_reduction_4k", "max_seq_length": 4096, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16}, {"name": "context_reduction_2k", "max_seq_length": 2048, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16}]}},
        "output": {"checkpoints": "artifacts/checkpoints"},
        "smoke_test": {"input": "smoke.jsonl", "max_seq_length": 2048, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1, "output_dir": "artifacts/smoke", "max_steps": 2},
    }


def test_load_and_format_chatml(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n", encoding="utf-8")
    records = load_chatml(path)
    assert len(load_chatml(path, offset=1, limit=1)) == 1

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            return "<|im_start|>assistant\n" + messages[-1]["content"] + "<|im_end|>\n"

    assert format_chatml(records, Tokenizer())[0]["text"].endswith("<|im_end|>\n")


def test_resolve_text_tokenizer_unwraps_multimodal_processor() -> None:
    class TextTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": []}

        def apply_chat_template(self, messages, **kwargs):
            return "rendered"

    class VisionProcessor:
        def __init__(self) -> None:
            self.tokenizer = TextTokenizer()

        def __call__(self, images=None, text=None, **kwargs):
            raise AssertionError("Vision processor must not receive text-only training records")

    processor = VisionProcessor()
    assert resolve_text_tokenizer(processor) is processor.tokenizer
    assert resolve_text_tokenizer(processor.tokenizer) is processor.tokenizer

    with pytest.raises(PipelineError, match="usable text tokenizer"):
        resolve_text_tokenizer(object())


def test_token_length_filter_excludes_without_silent_truncation() -> None:
    records = [
        {"text": "abc", "meta": {"content_hash": "short", "category": "concepto", "source": "unit"}},
        {"text": "abcdefgh", "meta": {"content_hash": "long", "category": "troubleshooting", "source": "unit"}},
    ]

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {"add_special_tokens": False, "truncation": False}
            return {"input_ids": [list(range(len(text))) for text in texts]}

    kept, report = filter_formatted_by_token_length(
        records, Tokenizer(), max_length=5, batch_size=1
    )

    assert [record["meta"]["content_hash"] for record in kept] == ["short"]
    assert report["input_records"] == 2
    assert report["kept_records"] == 1
    assert report["excluded_records"] == 1
    assert report["excluded"] == [{
        "content_hash": "long",
        "category": "troubleshooting",
        "source": "unit",
        "tokens": 8,
    }]


def test_resolve_model_requires_complete_local_weights(tmp_path: Path) -> None:
    config = _config()
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json", "model-1.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    assert resolve_model_reference(config, tmp_path) == str(model.resolve())
    (model / "model-1.safetensors").write_bytes(b"changed")
    with pytest.raises(PipelineError, match="hash mismatch"):
        verify_local_model_provenance(model, config["model"])


def test_local_model_git_revision_uses_command_scoped_safe_directory(
    monkeypatch, tmp_path: Path
) -> None:
    config = _config()
    model = tmp_path / "model"
    model.mkdir()
    (model / ".git").mkdir()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-1.safetensors"}}), encoding="utf-8"
    )
    (model / "model-1.safetensors").write_bytes(b"{}")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        return train_core.subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")

    monkeypatch.setattr(train_core.subprocess, "run", fake_run)

    result = verify_local_model_provenance(model, config["model"])

    resolved = model.resolve()
    assert result["revision"] == "a" * 40
    assert calls == [[
        "git", "-c", f"safe.directory={resolved}", "-C", str(resolved), "rev-parse", "HEAD"
    ]]


def test_attempts_and_latest_checkpoint_are_deterministic(tmp_path: Path) -> None:
    config = _config()
    primary = build_attempt(config, tmp_path, smoke_test=False)
    fallback_4k = build_attempt(config, tmp_path, smoke_test=False, fallback_index=0)
    fallback_2k = build_attempt(config, tmp_path, smoke_test=False, fallback_index=1)
    assert primary.effective_batch_size == 16
    assert fallback_4k.max_seq_length == 4096
    assert fallback_2k.max_seq_length == 2048
    assert primary.output_dir.name == "primary"
    assert fallback_4k.output_dir.name == "context_reduction_4k"
    assert fallback_2k.output_dir.name == "context_reduction_2k"
    assert len({primary.output_dir, fallback_4k.output_dir, fallback_2k.output_dir}) == 3
    output = tmp_path / "artifacts" / "checkpoints"
    checkpoint_9 = output / "checkpoint-9"
    checkpoint_9.mkdir(parents=True)
    (checkpoint_9 / "trainer_state.json").write_text(
        json.dumps({"global_step": 9}), encoding="utf-8"
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint_9 / name).write_bytes(b"complete")
    checkpoint_100 = output / "checkpoint-100"
    checkpoint_100.mkdir()
    assert latest_checkpoint(output).endswith("checkpoint-9")
    (checkpoint_100 / "trainer_state.json").write_text(
        json.dumps({"global_step": 100}), encoding="utf-8"
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint_100 / name).write_bytes(b"complete")
    assert latest_checkpoint(output).endswith("checkpoint-100")
    (checkpoint_9 / "checkpoint_provenance.json").write_text(
        json.dumps({"fingerprint": "current"}), encoding="utf-8"
    )
    (checkpoint_100 / "checkpoint_provenance.json").write_text(
        json.dumps({"fingerprint": "stale"}), encoding="utf-8"
    )
    assert latest_checkpoint(output, expected_fingerprint="current").endswith("checkpoint-9")
    assert latest_checkpoint(output, expected_fingerprint="missing") is None


def test_checkpoint_callback_persists_training_contract(monkeypatch, tmp_path: Path) -> None:
    class TrainerCallback:
        pass

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(TrainerCallback=TrainerCallback)
    )
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    callback = build_checkpoint_provenance_callback(
        {"version": 1, "training_profile": {"name": "primary"}},
        fingerprint="fingerprint-12",
    )
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=12),
        SimpleNamespace(),
    )
    provenance = json.loads(
        (checkpoint / "checkpoint_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["fingerprint"] == "fingerprint-12"
    assert provenance["contract"]["training_profile"]["name"] == "primary"


def test_warmup_steps_cover_smoke_and_full_training() -> None:
    assert _warmup_steps(
        examples=8,
        epochs=3,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        warmup_ratio=0.03,
    ) == 1
    assert _warmup_steps(
        examples=19_612,
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=-1,
        warmup_ratio=0.03,
    ) == 111


def test_training_config_rejects_manual_targets() -> None:
    config = _config()
    validate_training_config(config)
    config["lora"]["target_modules"] = ["q_proj"]
    with pytest.raises(PipelineError, match="hybrid attention"):
        validate_training_config(config)


def test_training_config_requires_random_sampling_for_batch_one() -> None:
    config = _config()
    config["trainer"]["train_sampling_strategy"] = "group_by_length"
    with pytest.raises(PipelineError, match="random sampling"):
        validate_training_config(config)


@pytest.mark.parametrize("field", ["per_device_eval_batch_size", "eval_steps"])
def test_training_config_requires_positive_evaluation_settings(field: str) -> None:
    config = _config()
    config["trainer"][field] = 0
    with pytest.raises(PipelineError, match=f"{field} must be positive"):
        validate_training_config(config)


def test_training_config_accepts_eval_loss_best_model_restoration() -> None:
    config = _config()
    config["trainer"].update({
        "eval_strategy": "steps",
        "save_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    })
    validate_training_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("metric_for_best_model", "loss", "select eval_loss"),
        ("greater_is_better", True, "greater_is_better=false"),
        ("save_steps", 300, "multiple of eval_steps"),
    ],
)
def test_training_config_rejects_invalid_best_model_contract(
    field: str, value: object, message: str,
) -> None:
    config = _config()
    config["trainer"].update({
        "eval_strategy": "steps",
        "save_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        field: value,
    })
    with pytest.raises(PipelineError, match=message):
        validate_training_config(config)


def test_training_config_rejects_non_reducing_oom_profile() -> None:
    config = _config()
    config["trainer"]["oom_fallback"]["profiles"][0] = {
        "name": "not_a_fallback",
        "max_seq_length": 8192,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
    }
    with pytest.raises(PipelineError, match="reduce memory monotonically"):
        validate_training_config(config)


def test_inspect_local_qwen_model(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "<|im_start|>"}), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"weights")
    report = inspect_local_model(
        tmp_path,
        instruction_delimiter="<|im_start|>user\n",
        response_delimiter="<|im_start|>assistant\n",
    )
    assert report["weights_bytes"] == 7


def test_full_training_requires_final_baseline_on_same_split(tmp_path: Path) -> None:
    config = {
        "prerequisites": {
            "require_final_baseline": True,
            "baseline_results": "benchmarks/baseline_results.json",
            "benchmark_config": "config/benchmark.yaml",
        }
    }
    statistics = {"outputs": {"test": {"sha256": "frozen-test"}}}
    benchmark_config = {
        "inputs": {"judge_manifest": "benchmarks/judge.jsonl"},
        "llm_judge": {"prompt": "benchmarks/judge.md"},
    }
    config_path = tmp_path / "config" / "benchmark.yaml"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(benchmark_config), encoding="utf-8")
    baseline_path = tmp_path / "benchmarks" / "baseline_results.json"
    baseline_path.parent.mkdir()
    (tmp_path / "benchmarks" / "judge.jsonl").write_text(
        '{"content_hash":"one"}\n', encoding="utf-8"
    )
    judge_prompt = tmp_path / "benchmarks" / "judge.md"
    judge_prompt.write_text("fixed rubric", encoding="utf-8")
    with pytest.raises(PipelineError, match="before full training"):
        _verify_baseline(config, tmp_path, statistics)
    baseline = {
        "variant": "baseline",
        "provisional": False,
        "split_test_sha256": "frozen-test",
        "completion": {"full_test": True, "generation": True, "syntax": True, "judge": True},
        "judge_provenance": judge_provenance(benchmark_config, tmp_path),
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert _verify_baseline(config, tmp_path, statistics) == baseline
    identity = {
        "alias": "gemma-4-12b-judge",
        "filename": "judge-Q6_K_XL.gguf",
        "quantization": "q6_k_xl",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    benchmark_config["llm_judge"]["model_identity"] = identity
    config_path.write_text(json.dumps(benchmark_config), encoding="utf-8")
    with pytest.raises(PipelineError, match="judge model identity"):
        _verify_baseline(config, tmp_path, statistics)
    baseline["judge_runtime"] = {
        "expected": identity,
        "served": {
            "alias": identity["alias"],
            "filename": identity["filename"],
            "quantization": identity["quantization"],
        },
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert _verify_baseline(config, tmp_path, statistics) == baseline
    judge_prompt.write_text("changed rubric", encoding="utf-8")
    with pytest.raises(PipelineError, match="judge prompt or manifest"):
        _verify_baseline(config, tmp_path, statistics)
    baseline["split_test_sha256"] = "other-test"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(PipelineError, match="frozen test split"):
        _verify_baseline(config, tmp_path, statistics)


def test_export_manifest_detects_artifact_changes(tmp_path: Path) -> None:
    merged = tmp_path / "artifacts" / "merged"
    merged.mkdir(parents=True)
    model_file = merged / "model.safetensors"
    model_file.write_bytes(b"weights")
    entries = artifact_files_manifest(merged, tmp_path)
    manifest = {
        "base_model": {"revision": "a" * 40},
        "training_split": {"test": "test-hash"},
        "artifacts": {"merged": entries},
    }
    manifest_path = tmp_path / "artifacts" / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_export_artifact(tmp_path, manifest_path, "merged")
    assert verified["file_count"] == 1
    model_file.write_bytes(b"changed")
    with pytest.raises(PipelineError, match="(size|hash) mismatch"):
        verify_export_artifact(tmp_path, manifest_path, "merged")


def test_finalize_gguf_export_relocates_and_requires_each_quantization(tmp_path: Path) -> None:
    merged = tmp_path / "artifacts" / "merged"
    exported = tmp_path / "artifacts" / "merged_gguf"
    configured = tmp_path / "artifacts" / "gguf"
    exported.mkdir(parents=True)
    q4 = exported / "Qwen3.5-4B-Q4_K_M.gguf"
    q8 = exported / "Qwen3.5-4B-Q8_0.gguf"
    q4.write_bytes(b"q4")
    q8.write_bytes(b"q8")

    result = finalize_gguf_export(
        {
            "gguf_directory": str(exported),
            "gguf_files": [str(q4), str(q8)],
        },
        configured_dir=configured,
        merged_dir=merged,
        root=tmp_path,
        required_quantizations=["q4_k_m", "q8_0"],
    )

    assert result["directory"] == configured
    assert not exported.exists()
    assert (configured / q4.name).read_bytes() == b"q4"
    assert (configured / q8.name).read_bytes() == b"q8"


def test_finalize_gguf_export_rejects_missing_quantization(tmp_path: Path) -> None:
    merged = tmp_path / "artifacts" / "merged"
    exported = tmp_path / "artifacts" / "merged_gguf"
    exported.mkdir(parents=True)
    q4 = exported / "Qwen3.5-4B-Q4_K_M.gguf"
    q4.write_bytes(b"q4")

    with pytest.raises(PipelineError, match="required GGUF quantizations"):
        finalize_gguf_export(
            {"gguf_directory": str(exported), "gguf_files": [str(q4)]},
            configured_dir=tmp_path / "artifacts" / "gguf",
            merged_dir=merged,
            root=tmp_path,
            required_quantizations=["q4_k_m", "q8_0"],
        )
    assert exported.is_dir()


def test_oom_fallback_runs_after_primary_exception_scope(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    config["output"]["metrics"] = "artifacts/metrics"
    calls = []

    def fake_train_once(config, root, attempt, *, export, resume_from_latest=False):
        calls.append(attempt)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"status": "trained"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_runner, "load_yaml", lambda path: config)
    monkeypatch.setattr(train_runner, "_train_once", fake_train_once)
    result = train_runner.run_training(smoke_test=False, export=False)
    assert len(calls) == 2
    assert calls[1].max_seq_length == 4096
    assert calls[1].batch_size == 1
    assert result["oom_fallback_used"] is True
    assert result["oom_failed_profiles"] == ["primary"]


def test_smoke_profile_is_not_reported_as_oom_fallback(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    config["output"]["metrics"] = "artifacts/metrics"

    def fake_train_once(config, root, attempt, *, export, resume_from_latest=False):
        return {"status": "trained", "training_profile": attempt.profile}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_runner, "load_yaml", lambda path: config)
    monkeypatch.setattr(train_runner, "_train_once", fake_train_once)
    result = train_runner.run_training(smoke_test=True, export=False)

    assert result["training_profile"] == "smoke"
    assert "oom_fallback_used" not in result
    assert "oom_failed_profiles" not in result


def test_oom_fallback_reduces_context_only_after_batch_one_oom(monkeypatch, tmp_path: Path) -> None:
    config = _config()
    config["output"]["metrics"] = "artifacts/metrics"
    calls = []

    def fake_train_once(config, root, attempt, *, export, resume_from_latest=False):
        calls.append(attempt)
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory")
        return {"status": "trained", "training_profile": attempt.profile}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_runner, "load_yaml", lambda path: config)
    monkeypatch.setattr(train_runner, "_train_once", fake_train_once)
    result = train_runner.run_training(smoke_test=False, export=False)

    assert [attempt.max_seq_length for attempt in calls] == [8192, 4096, 2048]
    assert [attempt.batch_size for attempt in calls] == [1, 1, 1]
    assert result["training_profile"] == "context_reduction_2k"
    assert result["oom_failed_profiles"] == ["primary", "context_reduction_4k"]


def test_oom_fallback_resumes_after_power_loss_without_retrying_failed_profile(
    monkeypatch, tmp_path: Path,
) -> None:
    config = _config()
    config["output"]["metrics"] = "artifacts/metrics"
    state_path = tmp_path / "artifacts" / "metrics" / "oom_state.json"
    contract = train_runner._oom_state_contract(config, tmp_path)
    train_runner._write_oom_state(
        state_path, contract=contract, failed_profiles=["primary"]
    )
    calls = []

    def fake_train_once(config, root, attempt, *, export, resume_from_latest=False):
        calls.append(attempt)
        return {"status": "trained", "training_profile": attempt.profile}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_runner, "load_yaml", lambda path: config)
    monkeypatch.setattr(train_runner, "_train_once", fake_train_once)
    result = train_runner.run_training(smoke_test=False, export=False)

    assert [attempt.profile for attempt in calls] == ["context_reduction_4k"]
    assert result["oom_fallback_used"] is True
    assert result["oom_failed_profiles"] == ["primary"]
    assert not state_path.exists()


def test_rocm_profile_rejects_quantization_and_checkpoint_reuse() -> None:
    config = _config()
    config["runtime"] = {"accelerator": "rocm", "expected_gpu_name": "AMD Radeon AI PRO R9700"}
    config["model"].update({"load_in_4bit": False, "dtype": "bfloat16"})
    config["trainer"].update({"optimizer": "adamw_torch", "resume_from_checkpoint": "none"})
    config["output"].update({
        "checkpoints": "artifacts/rocm/checkpoints",
        "adapter": "artifacts/rocm/adapter",
        "merged": "artifacts/rocm/merged",
        "gguf": "artifacts/rocm/gguf",
        "metrics": "artifacts/rocm/metrics",
        "manifest": "artifacts/rocm/export_manifest.json",
    })
    validate_training_config(config)
    config["model"]["load_in_4bit"] = True
    with pytest.raises(PipelineError, match="unquantized BF16"):
        validate_training_config(config)
    config["model"]["load_in_4bit"] = False
    config["trainer"]["resume_from_checkpoint"] = "auto"
    with pytest.raises(PipelineError, match="without automatic checkpoint resume"):
        validate_training_config(config)


def test_rocm_profile_requires_local_base_model(tmp_path: Path) -> None:
    config = _config()
    config["runtime"] = {"require_local_model": True}
    with pytest.raises(PipelineError, match="complete local base model"):
        resolve_model_reference(config, tmp_path)


def test_rocm_validation_redaction_is_exactly_pinned(tmp_path: Path) -> None:
    from docker_k8s_finetune.dedupe.exact import canonical_content
    from docker_k8s_finetune.io import content_hash, file_sha256

    original = _record()
    original["meta"]["source_record_id"] = "one-answer"
    original["meta"]["content_hash"] = content_hash(canonical_content(original))
    redacted = json.loads(json.dumps(original))
    redacted["messages"][1]["content"] = "Pregunta con secreto redactado"
    actual_content_hash = content_hash(canonical_content(redacted))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename, record in (("train.jsonl", original), ("val.jsonl", redacted), ("test.jsonl", original)):
        (data_dir / filename).write_text(json.dumps(record) + "\n", encoding="utf-8")
    statistics = {"provisional": False, "outputs": {
        "train": {"sha256": file_sha256(data_dir / "train.jsonl")},
        "validation": {"sha256": "historical-manifest-hash", "bytes": 100},
        "test": {"sha256": file_sha256(data_dir / "test.jsonl")},
    }}
    (data_dir / "statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
    config = {
        "runtime": {"accelerator": "rocm"},
        "data": {
            "train": "data/train.jsonl", "validation": "data/val.jsonl",
            "test": "data/test.jsonl", "split_statistics": "data/statistics.json",
            "validation_redaction": {
                "manifest_sha256": "historical-manifest-hash", "manifest_bytes": 100,
                "actual_sha256": file_sha256(data_dir / "val.jsonl"),
                "actual_bytes": (data_dir / "val.jsonl").stat().st_size,
                "record_line": 1, "source_record_id": "one-answer",
                "recorded_content_hash": original["meta"]["content_hash"],
                "redacted_content_hash": actual_content_hash,
            },
        },
    }
    verified = train_runner._verify_final_split(config, tmp_path)
    assert verified["outputs"]["validation"]["sha256"] == file_sha256(data_dir / "val.jsonl")
    assert verified["outputs"]["validation"]["manifest_sha256"] == "historical-manifest-hash"
    assert statistics["outputs"]["validation"]["sha256"] == "historical-manifest-hash"

    config["data"]["validation_redaction"]["redacted_content_hash"] = "wrong"
    with pytest.raises(PipelineError, match="record hashes"):
        train_runner._verify_final_split(config, tmp_path)


def test_rocm_math_library_provenance_pins_patched_binaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPBLASLT_TENSILE_LIBPATH", raising=False)
    with pytest.raises(PipelineError, match="requires a patched hipBLASLt overlay"):
        train_runner._rocm_math_library_provenance(required=True)
    monkeypatch.setenv("HIPBLASLT_TENSILE_LIBPATH", str(tmp_path))
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "roundup_power2_divisions:16")
    for index in range(4):
        stem = f"TensileLibrary_BB_BB_test{index}_gfx1201"
        (tmp_path / f"{stem}.co").write_bytes(f"object-{index}".encode())
        (tmp_path / f"{stem}.dat").write_bytes(f"logic-{index}".encode())

    provenance = train_runner._rocm_math_library_provenance()
    assert provenance["environment"]["PYTORCH_HIP_ALLOC_CONF"] == "roundup_power2_divisions:16"
    assert len(provenance["patched_files_sha256"]) == 8
    name = "TensileLibrary_BB_BB_test0_gfx1201.co"
    assert provenance["patched_files_sha256"][name] == hashlib.sha256(b"object-0").hexdigest()

    (tmp_path / name).unlink()
    (tmp_path / name).symlink_to(tmp_path / "TensileLibrary_BB_BB_test1_gfx1201.co")
    with pytest.raises(PipelineError, match="points to stock"):
        train_runner._rocm_math_library_provenance()


def test_rocm_resume_flag_is_explicit_and_uses_only_primary(monkeypatch, tmp_path: Path) -> None:
    config = train_runner.load_yaml(Path(__file__).resolve().parents[1] / "config/training.rocm.patched.yaml")
    calls = []

    def fake_train_once(config, root, attempt, *, export, resume_from_latest=False):
        calls.append((attempt.profile, resume_from_latest))
        return {"status": "trained"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_runner, "load_yaml", lambda path: config)
    monkeypatch.setattr(train_runner, "_train_once", fake_train_once)
    train_runner.run_training(export=False, resume_from_latest=True)
    assert calls == [("primary", True)]
    assert config["trainer"]["resume_from_checkpoint"] == "none"

    with pytest.raises(PipelineError, match="full training run"):
        train_runner.run_training(smoke_test=True, resume_from_latest=True)
    with pytest.raises(PipelineError, match="full training run"):
        train_runner.run_training(preflight_only=True, resume_from_latest=True)
