from __future__ import annotations

import json
from pathlib import Path

import pytest

from docker_k8s_finetune.benchmark.core import (
    aggregate_results,
    exact_match,
    judge_provenance,
    parse_judge_payload,
    prompt_messages,
    syntax_provenance,
)
from docker_k8s_finetune.benchmark.report import build_markdown
from docker_k8s_finetune.benchmark import pipeline as benchmark_pipeline
from docker_k8s_finetune.benchmark import report as benchmark_report
from docker_k8s_finetune.benchmark.backends import (
    Generation,
    UnslothBackend,
    _completion_length_through_first_eos,
    _cuda_runtime_metadata,
    _verify_llama_props,
)
from docker_k8s_finetune.benchmark.scoring import _verify_judge_props
from docker_k8s_finetune.benchmark.pipeline import _is_provisional
from docker_k8s_finetune.benchmark.pipeline import _generation_cache_fingerprint
from docker_k8s_finetune.errors import PipelineError
from docker_k8s_finetune.io import file_sha256


def _result(value: float) -> dict:
    metric = {"count": 1, "mean": value, "stddev": 0.0, "ci95_low": value, "ci95_high": value}
    categories = {
        category: {
            "exact_match": metric,
            "semantic_similarity": metric,
            "llm_judge": metric,
            "syntax_validity": metric,
        }
        for category in ("concepto", "comando_cli", "generacion_yaml", "troubleshooting", "dockerfile", "arquitectura", "out_of_domain")
    }
    return {
        "created_at": "2026-08-04T00:00:00Z",
        "runtime": {
            "backend": "unsloth",
            "device": "NVIDIA GeForce RTX 5060 Ti",
            "accelerator": "cuda",
            "compute_capability": "12.0",
            "torch_version": "2.10.0+cu128",
            "torch_cuda_version": "12.8",
        },
        "metrics": {"by_category": categories},
    }


def test_exact_match_unwraps_shell_fence() -> None:
    assert exact_match("kubectl get pods", " " + chr(96) * 3 + "bash\nkubectl get pods\n" + chr(96) * 3) == 1.0


def test_batched_completion_count_excludes_padding_after_first_eos() -> None:
    assert _completion_length_through_first_eos([10, 11, 2, 2, 2], {2}) == 3
    assert _completion_length_through_first_eos([10, 11], {2}) == 2


def test_generation_at_token_limit_is_not_retried_after_eos() -> None:
    class Backend:
        calls = 0

        def generate(self, messages, generation):
            self.calls += 1
            return Generation(
                text="complete",
                latency_seconds=1.0,
                prompt_tokens=2,
                completion_tokens=8,
                finished_eos=True,
            )

    backend = Backend()
    result = benchmark_pipeline._generate_with_truncation_retry(
        backend,
        [{"role": "user", "content": "question"}],
        {
            "max_new_tokens": 8,
            "retry_on_truncation": {
                "enabled": True,
                "repetition_penalty": 1.1,
            },
        },
    )

    assert backend.calls == 1
    assert result["finished_eos"] is True
    assert result["retry_count"] == 0


def test_unsloth_batch_requires_versioned_padding_policy() -> None:
    with pytest.raises(PipelineError, match="batch_policy"):
        UnslothBackend.generate_batch(object(), [[], []], {"seed": 3407})


def test_judge_scores_are_strictly_bounded() -> None:
    assert parse_judge_payload({"correctness": 5, "completeness": 3})["mean"] == 4.0
    with pytest.raises(PipelineError, match=r"\[1, 5\]"):
        parse_judge_payload({"correctness": 6, "completeness": 3})


def test_judge_props_require_fixed_alias_file_and_quantization() -> None:
    identity = {
        "alias": "gemma-4-12b-judge",
        "filename": "gemma-4-12b-it-UD-Q6_K_XL.gguf",
        "quantization": "q6_k_xl",
        "bytes": 10685011360,
        "sha256": "e" * 64,
    }
    verified = _verify_judge_props(
        {
            "model_alias": "gemma-4-12b-judge",
            "model_path": r"G:\models\gemma-4-12b-it-UD-Q6_K_XL.gguf",
            "model_ftype": "Q6_K",
        },
        expected_identity=identity,
    )
    assert verified["filename"] == identity["filename"]
    with pytest.raises(PipelineError, match="alias mismatch"):
        _verify_judge_props(
            {
                "model_alias": "other",
                "model_path": r"G:\models\gemma-4-12b-it-UD-Q6_K_XL.gguf",
            },
            expected_identity=identity,
        )
    with pytest.raises(PipelineError, match="model mismatch"):
        _verify_judge_props(
            {
                "model_alias": "gemma-4-12b-judge",
                "model_path": r"G:\models\other-Q6_K_XL.gguf",
            },
            expected_identity=identity,
        )


def test_benchmark_response_instruction_extends_existing_system_prompt() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "Domain expert."},
            {"role": "user", "content": "Create a manifest."},
            {"role": "assistant", "content": "reference"},
        ]
    }

    messages = prompt_messages(record, "Be concise and return complete code.")

    assert messages == [
        {
            "role": "system",
            "content": "Domain expert.\n\nBe concise and return complete code.",
        },
        {"role": "user", "content": "Create a manifest."},
    ]


def test_benchmark_response_instruction_adds_missing_system_prompt() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "Explain a pod."},
            {"role": "assistant", "content": "reference"},
        ]
    }

    messages = prompt_messages(record, "Answer in the question language.")

    assert messages[0] == {
        "role": "system",
        "content": "Answer in the question language.",
    }
    assert messages[1] == {"role": "user", "content": "Explain a pod."}


def test_aggregate_reports_category_metrics() -> None:
    record = {
        "category": "comando_cli", "error": None, "exact_match": 1.0,
        "semantic_similarity": 0.8, "syntax_valid": None,
        "judge": {"mean": 4.0}, "latency_seconds": 2.0, "tokens_per_second": 20.0,
        "truncated": False,
    }
    summary = aggregate_results([record])
    assert summary["by_category"]["comando_cli"]["semantic_similarity"]["mean"] == 0.8
    assert summary["overall"]["truncation_rate"]["mean"] == 0.0
    assert summary["overall"]["retry_rate"]["mean"] == 0.0


def test_report_calls_out_regressions() -> None:
    markdown = build_markdown(_result(0.8), _result(0.7), _result(0.69), loss_chart_path=None)
    assert "regresiones" in markdown
    assert "concepto" in markdown
    assert "Incertidumbre del LLM-juez" in markdown
    assert "Rendimiento en el hardware evaluado" in markdown
    assert "Integridad de la generacion" in markdown
    assert "Exact match de comandos CLI" in markdown
    assert "Exact safetensors" in markdown
    assert "Conservacion fuera de dominio" in markdown
    assert "NVIDIA GeForce RTX 5060 Ti (CUDA sm_120)" in markdown
    assert "Sintaxis GGUF" in markdown
    assert "cuantizacion concepto" in markdown
    assert "fuera de dominio: similitud semantica" in markdown


def test_report_recomputes_metrics_and_requires_frozen_record_order(tmp_path) -> None:
    (tmp_path / "test_manifest.jsonl").write_text(
        '{"content_hash":"test-one"}\n', encoding="utf-8"
    )
    (tmp_path / "ood.jsonl").write_text(
        '{"meta":{"content_hash":"ood-one"}}\n', encoding="utf-8"
    )
    records = [
        {
            "content_hash": content_hash,
            "category": category,
            "error": None,
            "exact_match": 1.0,
            "semantic_similarity": 0.8,
            "syntax_valid": None,
            "judge": {"mean": 4.0},
            "truncated": False,
            "retry_count": 0,
            "generation_attempts": 1,
            "generated_completion_tokens_total": 10,
            "latency_seconds": 1.0,
            "tokens_per_second": 10.0,
        }
        for content_hash, category in (
            ("test-one", "concepto"),
            ("ood-one", "out_of_domain"),
        )
    ]
    results = {
        variant: {
            "records": json.loads(json.dumps(records)),
            "metrics": aggregate_results(records),
        }
        for variant in ("baseline", "finetuned_safetensors", "finetuned_gguf")
    }
    config = {
        "inputs": {
            "test_manifest": "test_manifest.jsonl",
            "out_of_domain": "ood.jsonl",
        }
    }
    benchmark_report._validate_frozen_record_coverage(results, config, tmp_path)

    results["finetuned_gguf"]["records"].reverse()
    with pytest.raises(PipelineError, match="frozen test plus OOD order"):
        benchmark_report._validate_frozen_record_coverage(results, config, tmp_path)
    results["finetuned_gguf"]["records"].reverse()
    results["finetuned_gguf"]["metrics"]["overall"]["count"] = 999
    with pytest.raises(PipelineError, match="record-level evidence"):
        benchmark_report._validate_frozen_record_coverage(results, config, tmp_path)


def test_report_reverifies_current_export_artifacts(tmp_path) -> None:
    merged = tmp_path / "artifacts" / "merged" / "model.safetensors"
    gguf = tmp_path / "artifacts" / "gguf" / "model-Q4_K_M.gguf"
    merged.parent.mkdir(parents=True)
    gguf.parent.mkdir(parents=True)
    merged.write_bytes(b"merged")
    gguf.write_bytes(b"gguf")
    manifest_path = tmp_path / "artifacts" / "export_manifest.json"
    manifest = {
        "base_model": {"revision": "a" * 40},
        "training_split": {
            "train": "train-hash",
            "validation": "validation-hash",
            "test": "split-hash",
        },
        "training_length_filter": {"action": "exclude", "train": {}, "validation": {}},
        "training_contract": {
            "fingerprint": "contract-hash",
            "contract": {"version": 1},
        },
        "artifacts": {
            "merged": [{
                "path": "artifacts/merged/model.safetensors",
                "bytes": merged.stat().st_size,
                "sha256": file_sha256(merged),
            }],
            "gguf": [{
                "path": "artifacts/gguf/model-Q4_K_M.gguf",
                "bytes": gguf.stat().st_size,
                "sha256": file_sha256(gguf),
            }],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "variants": {
            "finetuned_safetensors": {
                "export_manifest": "artifacts/export_manifest.json",
                "artifact": "merged",
            },
            "finetuned_gguf": {
                "export_manifest": "artifacts/export_manifest.json",
                "artifact": "gguf",
                "require_llama_props": True,
                "expected_quantization": "q4_k_m",
            },
        }
    }
    results = {
        "finetuned_safetensors": {
            "model_provenance": benchmark_report.verify_export_artifact(
                tmp_path, Path("artifacts/export_manifest.json"), "merged"
            )
        },
        "finetuned_gguf": {},
    }
    gguf_provenance = benchmark_report.verify_export_artifact(
        tmp_path, Path("artifacts/export_manifest.json"), "gguf"
    )
    served = {
        "model_path": "artifacts/gguf/model-Q4_K_M.gguf",
        "sha256": file_sha256(gguf),
        "quantization": "q4_k_m",
    }
    results["finetuned_gguf"] = {
        "model_provenance": {**gguf_provenance, "served_model": served},
        "runtime": {"served_model": served},
    }
    benchmark_report._validate_export_provenance(results, config, tmp_path)
    lineage_results = {
        variant: {"split_test_sha256": "split-hash"}
        for variant in ("baseline", "finetuned_safetensors", "finetuned_gguf")
    }
    training_summary = {
        "base_model": manifest["base_model"],
        "training_split": manifest["training_split"],
        "length_filter": manifest["training_length_filter"],
        "checkpoint_fingerprint": "contract-hash",
        "checkpoint_contract": {"version": 1},
    }
    benchmark_report._validate_training_export_lineage(
        training_summary, lineage_results, config, tmp_path
    )
    training_summary["training_split"] = {**manifest["training_split"], "test": "wrong"}
    with pytest.raises(PipelineError, match="different data splits"):
        benchmark_report._validate_training_export_lineage(
            training_summary, lineage_results, config, tmp_path
        )
    training_summary["training_split"] = manifest["training_split"]
    results["finetuned_gguf"]["runtime"]["served_model"] = {**served, "sha256": "wrong"}
    with pytest.raises(PipelineError, match="consistent served-model identity"):
        benchmark_report._validate_export_provenance(results, config, tmp_path)
    results["finetuned_gguf"]["runtime"]["served_model"] = served
    gguf.write_bytes(b"changed")
    with pytest.raises(PipelineError, match="(size|hash) mismatch"):
        benchmark_report._validate_export_provenance(results, config, tmp_path)


def test_final_report_uses_persisted_complete_loss_history(monkeypatch, tmp_path) -> None:
    config = {
        "outputs": {
            "baseline": "benchmarks/baseline.json",
            "finetuned_safetensors": "benchmarks/fine.json",
            "finetuned_gguf": "benchmarks/gguf.json",
            "comparison_report": "benchmarks/report.md",
            "loss_chart": "benchmarks/training_loss.svg",
        }
    }
    for key in ("baseline", "finetuned_safetensors", "finetuned_gguf"):
        result = _result(0.8)
        result.update({
            "variant": key,
            "provisional": False,
            "split_test_sha256": "frozen",
            "completion": {
                "full_test": True,
                "generation": True,
                "syntax": True,
                "judge": True,
            },
        })
        path = tmp_path / config["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
    training_result = tmp_path / "artifacts" / "metrics" / "training_result.json"
    training_result.parent.mkdir(parents=True)
    training_result.write_text(
        json.dumps({
            "max_seq_length": 8192,
            "batch_size": 4,
            "effective_batch_size": 16,
            "metrics": {"train_loss": 1.0},
            "evaluation_metrics": {"eval_loss": 1.1},
            "vram": {"peak_allocated_gib": 7.0, "peak_reserved_gib": 8.0},
            "length_filter": {
                "train": {
                    "input_records": 10,
                    "excluded_records": 1,
                    "kept_content_hashes_sha256": "train-hash",
                },
                "validation": {
                    "input_records": 4,
                    "excluded_records": 0,
                    "kept_content_hashes_sha256": "validation-hash",
                },
            },
            "log_history": [
                {"step": 1, "loss": 1.2},
                {"step": 1, "eval_loss": 1.3},
                {"step": 2, "loss": 1.0},
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(benchmark_report, "load_yaml", lambda path: config)

    output = benchmark_report.run_report(root=tmp_path)

    assert (tmp_path / "benchmarks" / "training_loss.svg").is_file()
    assert "training_loss.svg" in (tmp_path / "benchmarks" / "report.md").read_text(encoding="utf-8")
    assert "Ejemplos excluidos por longitud" in (tmp_path / "benchmarks" / "report.md").read_text(encoding="utf-8")
    assert output["loss_chart"].endswith("training_loss.svg")

    fine_path = tmp_path / config["outputs"]["finetuned_safetensors"]
    fine_result = json.loads(fine_path.read_text(encoding="utf-8"))
    fine_result["runtime"]["device"] = "different GPU"
    fine_path.write_text(json.dumps(fine_result), encoding="utf-8")
    with pytest.raises(PipelineError, match="different accelerator runtimes"):
        benchmark_report.run_report(root=tmp_path)
    fine_result["runtime"]["device"] = "NVIDIA GeForce RTX 5060 Ti"
    fine_path.write_text(json.dumps(fine_result), encoding="utf-8")

    training_result.unlink()
    with pytest.raises(PipelineError, match="training_result.json"):
        benchmark_report.run_report(root=tmp_path)

    training_result.write_text(
        json.dumps({"log_history": [{"step": 1, "loss": 1.2}]}),
        encoding="utf-8",
    )
    with pytest.raises(PipelineError, match="loss history"):
        benchmark_report.run_report(root=tmp_path)


def test_cuda_runtime_metadata_records_real_measurement_context() -> None:
    class Properties:
        name = "NVIDIA GeForce RTX 5060 Ti"
        total_memory = 16 * 1024**3
        major = 12
        minor = 0

    class Cuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return Properties()

    class Version:
        cuda = "12.8"

    class Torch:
        __version__ = "2.10.0"
        cuda = Cuda()
        version = Version()

    runtime = _cuda_runtime_metadata(Torch())
    assert runtime["device"] == "NVIDIA GeForce RTX 5060 Ti"
    assert runtime["total_memory_bytes"] == 16 * 1024**3
    assert runtime["compute_capability"] == "12.0"
    assert runtime["torch_cuda_version"] == "12.8"


def test_benchmark_completion_requires_every_stage() -> None:
    complete = {"full_test": True, "generation": True, "syntax": True, "judge": True}
    assert not _is_provisional({"provisional": False}, complete)
    assert _is_provisional({"provisional": False}, {**complete, "judge": False})
    assert _is_provisional({"provisional": True}, complete)


def test_judge_provenance_covers_all_frozen_benchmark_inputs(tmp_path) -> None:
    for name, content in (
        ("judge.jsonl", '{"content_hash":"one"}\n'),
        ("judge.md", "rubric"),
        ("test.jsonl", '{"content_hash":"one"}\n'),
        ("ood.jsonl", '{"meta":{"content_hash":"ood-1"}}\n'),
    ):
        (tmp_path / name).write_text(content, encoding="utf-8")
    config = {
        "inputs": {
            "judge_manifest": "judge.jsonl",
            "test_manifest": "test.jsonl",
            "out_of_domain": "ood.jsonl",
        },
        "llm_judge": {"prompt": "judge.md"},
    }
    first = judge_provenance(config, tmp_path)
    assert set(first) == {"manifest", "prompt", "test_manifest", "out_of_domain"}
    (tmp_path / "ood.jsonl").write_text(
        '{"meta":{"content_hash":"ood-2"}}\n', encoding="utf-8"
    )
    second = judge_provenance(config, tmp_path)
    assert first["out_of_domain"]["sha256"] != second["out_of_domain"]["sha256"]


def test_syntax_provenance_tracks_pinned_validator_configuration(tmp_path) -> None:
    validator = tmp_path / "validation.yaml"
    validator.write_text("version: 1\n", encoding="utf-8")
    config = {
        "syntax_validation": {
            "enabled": True,
            "config": validator.name,
        }
    }
    first = syntax_provenance(config, tmp_path)
    validator.write_text("version: 2\n", encoding="utf-8")
    second = syntax_provenance(config, tmp_path)
    assert first["path"] == "validation.yaml"
    assert first["sha256"] != second["sha256"]


def test_generation_cache_fingerprint_covers_prompt_and_backend() -> None:
    record = {"meta": {"content_hash": "one"}, "messages": [{"role": "user", "content": "hello"}]}
    first = _generation_cache_fingerprint(
        record, variant="baseline", backend={"model": "base"}, generation={"temperature": 0.0}
    )
    second = _generation_cache_fingerprint(
        record, variant="baseline", backend={"model": "other"}, generation={"temperature": 0.0}
    )
    assert first != second


def test_generation_cache_fingerprint_covers_batch_size() -> None:
    record = {"meta": {"content_hash": "one"}, "messages": [{"role": "user", "content": "hello"}]}
    first = _generation_cache_fingerprint(
        record,
        variant="baseline",
        backend={"model": "base"},
        generation={"temperature": 0.0, "batch_size": 1},
    )
    second = _generation_cache_fingerprint(
        record,
        variant="baseline",
        backend={"model": "base"},
        generation={"temperature": 0.0, "batch_size": 2},
    )
    assert first != second


def test_generation_cache_fingerprint_ignores_audit_only_threshold() -> None:
    record = {"meta": {"content_hash": "one"}, "messages": [{"role": "user", "content": "hello"}]}
    first = _generation_cache_fingerprint(
        record,
        variant="baseline",
        backend={"model": "base"},
        generation={
            "temperature": 0.0,
            "max_truncation_rate": 0.01,
            "cache_compatible_prior_max_new_tokens": [512],
        },
    )
    second = _generation_cache_fingerprint(
        record,
        variant="baseline",
        backend={"model": "base"},
        generation={
            "temperature": 0.0,
            "max_truncation_rate": 0.05,
            "cache_compatible_prior_max_new_tokens": [768],
        },
    )
    assert first == second


def test_saved_benchmark_can_complete_judging(monkeypatch, tmp_path) -> None:
    config = {
        "variants": {"baseline": {"backend": "test"}},
        "inputs": {"judge_manifest": "judge.jsonl"},
        "outputs": {
            "baseline": "baseline.json",
            "finetuned_safetensors": "fine.json",
            "finetuned_gguf": "gguf.json",
            "finetuned_combined": "combined.json",
        },
        "llm_judge": {"prompt": "judge.md"},
        "generation": {"max_new_tokens": 8},
    }
    (tmp_path / "judge.md").write_text("rubric", encoding="utf-8")
    (tmp_path / "judge.jsonl").write_text('{"content_hash":"one"}\n', encoding="utf-8")
    saved = {
        "variant": "baseline",
        "generation": config["generation"],
        "split_test_sha256": "split-hash",
        "completion": {
            "full_test": True,
            "generation": True,
            "syntax": True,
            "judge": False,
        },
        "provisional": True,
        "metrics": {},
        "judge_provenance": judge_provenance(config, tmp_path),
        "records": [{
            "content_hash": "one",
            "category": "concepto",
            "source": "unit",
            "question": "question",
            "reference": "reference",
            "prediction": "candidate",
            "latency_seconds": 1.0,
            "tokens_per_second": 10.0,
            "exact_match": None,
            "semantic_similarity": 0.5,
            "syntax_valid": None,
            "judge": {},
            "error": None,
        }],
    }
    (tmp_path / "baseline.json").write_text(json.dumps(saved), encoding="utf-8")
    expected = [{"meta": {"content_hash": "one"}}]
    statistics = {
        "provisional": False,
        "benchmark_manifests": {"judge": 1},
        "outputs": {"test": {"sha256": "split-hash", "count": 1}},
    }
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline, "_load_benchmark_records", lambda config, root, limit: (expected, statistics)
    )
    monkeypatch.setattr(benchmark_pipeline, "_load_manifest", lambda path: [{"content_hash": "one"}])

    class FakeJudge:
        def __init__(self, judge_config, prompt):
            assert prompt == "rubric"
            self.provenance = {"served": "judge-a"}

        def score(self, **kwargs):
            return {"correctness": 5.0, "completeness": 4.0, "mean": 4.5, "rationale": "ok"}

    monkeypatch.setattr(benchmark_pipeline, "JudgeClient", FakeJudge)
    result = benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)
    persisted = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert result["judged"] == 1
    assert result["cached"] == 0
    assert not result["provisional"]
    assert persisted["records"][0]["judge"]["mean"] == 4.5
    assert persisted["completion"]["judge"] is True

    class ReplacementJudge(FakeJudge):
        def __init__(self, judge_config, prompt):
            super().__init__(judge_config, prompt)
            self.provenance = {"served": "judge-b"}

    monkeypatch.setattr(benchmark_pipeline, "JudgeClient", ReplacementJudge)
    replaced = benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)
    assert replaced["judged"] == 1
    assert replaced["cached"] == 0

    (tmp_path / "judge.md").write_text("changed rubric", encoding="utf-8")
    with pytest.raises(PipelineError, match="different judge prompt or manifest"):
        benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)


def test_saved_benchmark_postprocessing_requires_successful_generation(monkeypatch, tmp_path) -> None:
    config = {
        "variants": {"baseline": {"backend": "test"}},
        "inputs": {"judge_manifest": "judge.jsonl"},
        "outputs": {"baseline": "baseline.json"},
        "generation": {"max_new_tokens": 8},
        "llm_judge": {"prompt": "judge.md"},
    }
    (tmp_path / "judge.md").write_text("rubric", encoding="utf-8")
    (tmp_path / "judge.jsonl").write_text('{"content_hash":"one"}\n', encoding="utf-8")
    saved = {
        "variant": "baseline",
        "generation": config["generation"],
        "split_test_sha256": "split-hash",
        "judge_provenance": judge_provenance(config, tmp_path),
        "completion": {
            "full_test": True,
            "generation": False,
            "syntax": False,
            "judge": False,
        },
        "records": [{"content_hash": "one"}],
    }
    (tmp_path / "baseline.json").write_text(json.dumps(saved), encoding="utf-8")
    expected = [{"meta": {"content_hash": "one"}}]
    statistics = {
        "provisional": False,
        "benchmark_manifests": {"judge": 1},
        "outputs": {"test": {"sha256": "split-hash", "count": 1}},
    }
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline, "_load_benchmark_records", lambda config, root, limit: (expected, statistics)
    )

    with pytest.raises(PipelineError, match="before successful generation"):
        benchmark_pipeline.run_benchmark_syntax(variant="baseline", root=tmp_path)
    with pytest.raises(PipelineError, match="before successful generation"):
        benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)


def test_judge_manifest_must_be_unique_and_within_frozen_test(tmp_path) -> None:
    config = {"inputs": {"judge_manifest": "judge.jsonl"}}
    expected = [
        {"meta": {"content_hash": "one", "category": "concepto", "source": "unit"}},
        {"meta": {"content_hash": "two", "category": "concepto", "source": "unit"}},
    ]
    statistics = {
        "benchmark_manifests": {"judge": 2},
        "outputs": {"test": {"count": 2}},
    }
    manifest = tmp_path / "judge.jsonl"
    manifest.write_text(
        '{"content_hash":"one"}\n{"content_hash":"one"}\n', encoding="utf-8"
    )
    with pytest.raises(PipelineError, match="non-empty and unique"):
        benchmark_pipeline._validated_judge_hashes(config, tmp_path, expected, statistics)

    manifest.write_text(
        '{"content_hash":"one"}\n{"content_hash":"unknown"}\n', encoding="utf-8"
    )
    with pytest.raises(PipelineError, match="outside the frozen held-out test"):
        benchmark_pipeline._validated_judge_hashes(config, tmp_path, expected, statistics)


def test_saved_benchmark_judge_resumes_after_interruption(monkeypatch, tmp_path) -> None:
    config = {
        "variants": {"baseline": {"backend": "test"}},
        "inputs": {"judge_manifest": "judge.jsonl"},
        "outputs": {
            "baseline": "baseline.json",
            "finetuned_safetensors": "fine.json",
            "finetuned_gguf": "gguf.json",
            "finetuned_combined": "combined.json",
        },
        "generation": {"max_new_tokens": 8},
        "llm_judge": {"prompt": "judge.md"},
    }
    (tmp_path / "judge.md").write_text("rubric", encoding="utf-8")
    (tmp_path / "judge.jsonl").write_text(
        '{"content_hash":"one"}\n{"content_hash":"two"}\n', encoding="utf-8"
    )
    records = []
    for content_hash in ("one", "two"):
        records.append({
            "content_hash": content_hash,
            "category": "concepto",
            "source": "unit",
            "question": f"question-{content_hash}",
            "reference": f"reference-{content_hash}",
            "prediction": f"candidate-{content_hash}",
            "latency_seconds": 1.0,
            "tokens_per_second": 10.0,
            "exact_match": None,
            "semantic_similarity": 0.5,
            "syntax_valid": None,
            "truncated": False,
            "judge": None,
            "error": None,
        })
    saved = {
        "variant": "baseline",
        "generation": config["generation"],
        "split_test_sha256": "split-hash",
        "completion": {
            "full_test": True,
            "generation": True,
            "syntax": True,
            "judge": False,
        },
        "provisional": True,
        "metrics": {},
        "judge_provenance": judge_provenance(config, tmp_path),
        "records": records,
    }
    (tmp_path / "baseline.json").write_text(json.dumps(saved), encoding="utf-8")
    expected = [{"meta": {"content_hash": value}} for value in ("one", "two")]
    statistics = {
        "provisional": False,
        "benchmark_manifests": {"judge": 2},
        "outputs": {"test": {"sha256": "split-hash", "count": 2}},
    }
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline, "_load_benchmark_records", lambda config, root, limit: (expected, statistics)
    )
    monkeypatch.setattr(
        benchmark_pipeline,
        "_load_manifest",
        lambda path: [{"content_hash": value} for value in ("one", "two")],
    )
    calls = []

    class InterruptingJudge:
        def __init__(self, judge_config, prompt):
            self.provenance = {"served": "fixed"}

        def score(self, **kwargs):
            calls.append(kwargs["question"])
            if len(calls) == 2:
                raise PipelineError("simulated outage")
            return {"correctness": 5.0, "completeness": 4.0, "mean": 4.5, "rationale": "ok"}

    monkeypatch.setattr(benchmark_pipeline, "JudgeClient", InterruptingJudge)
    with pytest.raises(PipelineError, match="simulated outage"):
        benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)

    interrupted = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert interrupted["records"][0]["judge"]["mean"] == 4.5
    assert interrupted["records"][1]["judge"] is None
    assert interrupted["completion"]["judge"] is False

    class ResumedJudge:
        def __init__(self, judge_config, prompt):
            self.provenance = {"served": "fixed"}

        def score(self, **kwargs):
            calls.append(kwargs["question"])
            return {"correctness": 4.0, "completeness": 4.0, "mean": 4.0, "rationale": "ok"}

    monkeypatch.setattr(benchmark_pipeline, "JudgeClient", ResumedJudge)
    resumed = benchmark_pipeline.run_benchmark_judge(variant="baseline", root=tmp_path)
    persisted = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert resumed["judged"] == 1
    assert resumed["cached"] == 1
    assert calls == ["question-one", "question-two", "question-two"]
    assert persisted["records"][1]["judge"]["mean"] == 4.0
    assert persisted["completion"]["judge"] is True
    assert persisted["provisional"] is False


def test_saved_benchmark_can_complete_host_syntax(monkeypatch, tmp_path) -> None:
    config = {
        "variants": {"baseline": {"backend": "test"}},
        "outputs": {"baseline": "baseline.json"},
        "generation": {"max_new_tokens": 8},
    }
    saved = {
        "variant": "baseline",
        "generation": config["generation"],
        "split_test_sha256": "split-hash",
        "completion": {
            "full_test": True,
            "generation": True,
            "syntax": False,
            "judge": False,
        },
        "provisional": True,
        "records": [{
            "content_hash": "one",
            "category": "dockerfile",
            "error": None,
            "syntax_valid": None,
            "syntax_reason": None,
            "exact_match": None,
            "semantic_similarity": 0.5,
            "latency_seconds": 1.0,
            "tokens_per_second": 10.0,
            "truncated": False,
            "judge": None,
        }],
    }
    (tmp_path / "baseline.json").write_text(json.dumps(saved), encoding="utf-8")
    expected = [{"meta": {"content_hash": "one"}}]
    statistics = {"provisional": False, "outputs": {"test": {"sha256": "split-hash"}}}
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline, "_load_benchmark_records", lambda config, root, limit: (expected, statistics)
    )
    monkeypatch.setattr(
        benchmark_pipeline,
        "syntax_scores",
        lambda records, **kwargs: {"one": {"valid": 1.0, "reason": None}},
    )

    result = benchmark_pipeline.run_benchmark_syntax(variant="baseline", root=tmp_path)
    persisted = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert result["checked"] == 1
    assert result["valid"] == 1
    assert persisted["records"][0]["syntax_valid"] == 1.0
    assert persisted["completion"]["syntax"] is True
    assert persisted["provisional"] is True  # The judge stage is still pending.

    monkeypatch.setattr(benchmark_pipeline, "syntax_scores", lambda records, **kwargs: {})
    with pytest.raises(PipelineError, match="exactly cover every YAML and Dockerfile"):
        benchmark_pipeline.run_benchmark_syntax(variant="baseline", root=tmp_path)


def test_llama_props_bind_benchmark_to_manifested_quantization(tmp_path) -> None:
    gguf = tmp_path / "artifacts" / "gguf" / "model-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    provenance = {
        "files": [{"path": "artifacts/gguf/model-Q4_K_M.gguf", "sha256": "abc"}]
    }
    verified = _verify_llama_props(
        {"model_alias": "qwen-docker-k8s", "model_path": str(gguf), "model_ftype": "Q4_K_M"},
        expected_alias="qwen-docker-k8s",
        expected_quantization="q4_k_m",
        root=tmp_path,
        provenance=provenance,
    )
    assert verified["sha256"] == "abc"
    assert verified["model_path"] == "artifacts/gguf/model-Q4_K_M.gguf"
    windows_verified = _verify_llama_props(
        {
            "model_alias": "qwen-docker-k8s",
            "model_path": r"D:\Archivos\project\artifacts\gguf\model-Q4_K_M.gguf",
            "model_ftype": "Q4_K_M",
        },
        expected_alias="qwen-docker-k8s",
        expected_quantization="q4_k_m",
        root=tmp_path,
        provenance=provenance,
    )
    assert windows_verified["server_model_path"].startswith("D:\\Archivos")
    with pytest.raises(PipelineError, match="alias mismatch"):
        _verify_llama_props(
            {"model_alias": "wrong", "model_path": str(gguf)},
            expected_alias="qwen-docker-k8s",
            expected_quantization="q4_k_m",
            root=tmp_path,
            provenance=provenance,
        )
    with pytest.raises(PipelineError, match="not the manifested GGUF"):
        _verify_llama_props(
            {
                "model_alias": "qwen-docker-k8s",
                "model_path": r"G:\models\model-Q4_K_M.gguf",
            },
            expected_alias="qwen-docker-k8s",
            expected_quantization="q4_k_m",
            root=tmp_path,
            provenance=provenance,
        )


def test_frozen_test_manifest_rejects_duplicate_hashes(tmp_path) -> None:
    test_path = tmp_path / "test.jsonl"
    manifest_path = tmp_path / "test_manifest.jsonl"
    statistics_path = tmp_path / "split_statistics.json"
    ood_path = tmp_path / "ood.jsonl"
    record = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "meta": {"content_hash": "one", "category": "concepto", "source": "unit"},
    }
    test_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest_path.write_text(
        '{"content_hash":"one"}\n{"content_hash":"one"}\n', encoding="utf-8"
    )
    ood_path.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "ood"},
                {"role": "assistant", "content": "answer"},
            ],
            "meta": {"content_hash": "ood-1", "category": "out_of_domain"},
        }) + "\n",
        encoding="utf-8",
    )
    statistics_path.write_text(
        json.dumps({"outputs": {"test": {"sha256": file_sha256(test_path)}}}),
        encoding="utf-8",
    )
    config = {
        "inputs": {
            "test": test_path.name,
            "split_statistics": statistics_path.name,
            "test_manifest": manifest_path.name,
            "out_of_domain": ood_path.name,
            "out_of_domain_count": 1,
        }
    }
    with pytest.raises(PipelineError, match="content hashes must be unique"):
        benchmark_pipeline._load_benchmark_records(config, tmp_path, limit=None)


def test_benchmark_generation_cache_is_reused(monkeypatch, tmp_path) -> None:
    config = {
        "version": 1,
        "variants": {"baseline": {"backend": "fake", "model": "base"}},
        "outputs": {"baseline": "baseline.json", "work_dir": "work"},
        "generation": {
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 8,
            "max_truncation_rate": 0.0,
        },
        "semantic_similarity": {},
        "syntax_validation": {"enabled": False},
        "llm_judge": {"enabled": False},
    }
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "meta": {"content_hash": "hash-one", "category": "comando_cli", "source": "unit"},
    }
    statistics = {"provisional": False, "outputs": {"test": {"sha256": "split"}}}

    class FakeBackend:
        provenance = {"model": "fake"}
        runtime = {"backend": "fake", "device": "unit GPU"}
        calls = 0

        def generate(self, messages, generation):
            self.calls += 1
            completion_tokens = 0 if generation.get("repetition_penalty") == 1.1 else 1
            return type("Generated", (), {
                "text": "answer",
                "latency_seconds": 1.0,
                "prompt_tokens": 2,
                "completion_tokens": completion_tokens,
                "tokens_per_second": float(completion_tokens),
            })()

    backend = FakeBackend()
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline, "_load_benchmark_records", lambda config, root, limit: ([record], statistics)
    )
    monkeypatch.setattr(benchmark_pipeline, "build_backend", lambda variant, root: backend)
    monkeypatch.setattr(benchmark_pipeline, "semantic_scores", lambda refs, preds, config: [1.0])
    first = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    second = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    assert backend.calls == 1
    assert first["generation_cache"]["generated"] == 1
    assert second["generation_cache"]["cached"] == 1
    assert second["runtime"]["device"] == "unit GPU"
    assert not second["provisional"]
    assert second["truncation_audit"] == {
        "max_allowed_rate": 0.0,
        "truncated": 0,
        "generated": 1,
        "rate": 0.0,
        "passed": True,
    }

    config["generation"]["max_new_tokens"] = 16
    config["generation"]["cache_compatible_prior_max_new_tokens"] = [8]
    migrated = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    assert backend.calls == 1
    assert migrated["generation_cache"]["compatible_reused"] == 1
    assert migrated["completion"]["generation"] is True

    config["generation"]["max_new_tokens"] = 1
    config["generation"]["cache_compatible_prior_max_new_tokens"] = []
    capped = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    assert backend.calls == 2
    assert capped["truncation_audit"]["passed"] is False
    assert capped["completion"]["generation"] is False
    assert capped["provisional"] is True

    config["generation"]["retry_on_truncation"] = {
        "enabled": True,
        "max_attempts": 1,
        "repetition_penalty": 1.1,
    }
    retried = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    assert backend.calls == 3  # The cached capped primary is not regenerated.
    assert retried["records"][0]["retry_count"] == 1
    assert retried["records"][0]["generation_attempts"] == 2
    assert retried["records"][0]["generated_completion_tokens_total"] == 1
    assert retried["records"][0]["truncated"] is False
    assert retried["completion"]["generation"] is True


def test_benchmark_prefills_cache_with_batched_backend(monkeypatch, tmp_path) -> None:
    config = {
        "version": 1,
        "variants": {"baseline": {"backend": "fake", "model": "base"}},
        "outputs": {"baseline": "baseline.json", "work_dir": "work"},
        "generation": {
            "seed": 3407,
            "batch_size": 2,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 8,
            "max_truncation_rate": 0.0,
        },
        "semantic_similarity": {},
        "syntax_validation": {"enabled": False},
        "llm_judge": {"enabled": False},
    }
    records = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"question-{index}"},
                {"role": "assistant", "content": f"answer-{index}"},
            ],
            "meta": {
                "content_hash": f"hash-{index}",
                "category": "comando_cli",
                "source": "unit",
            },
        }
        for index in range(2)
    ]
    statistics = {"provisional": False, "outputs": {"test": {"sha256": "split"}}}

    class FakeBatchBackend:
        provenance = {"model": "fake"}
        runtime = {"backend": "fake", "device": "unit GPU"}

        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def generate_batch(self, messages_batch, generation):
            self.batch_calls += 1
            assert len(messages_batch) == 2
            return [
                Generation(
                    text=f"answer-{index}",
                    latency_seconds=2.0,
                    prompt_tokens=3,
                    completion_tokens=index + 1,
                    batch_size=2,
                    batch_completion_tokens_total=3,
                )
                for index in range(2)
            ]

        def generate(self, messages, generation):
            self.single_calls += 1
            raise AssertionError("sequential fallback should not be used")

    backend = FakeBatchBackend()
    monkeypatch.setattr(benchmark_pipeline, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        benchmark_pipeline,
        "_load_benchmark_records",
        lambda config, root, limit: (records, statistics),
    )
    monkeypatch.setattr(benchmark_pipeline, "build_backend", lambda variant, root: backend)
    monkeypatch.setattr(benchmark_pipeline, "semantic_scores", lambda refs, preds, config: [1.0, 1.0])

    result = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)

    assert backend.batch_calls == 1
    assert backend.single_calls == 0
    assert result["generation_cache"]["generated"] == 2
    assert result["generation_cache"]["cached"] == 0
    assert [record["generation_batch_size"] for record in result["records"]] == [2, 2]
    assert result["metrics"]["overall"]["batch_tokens_per_second"]["mean"] == 1.5

    resumed = benchmark_pipeline.run_benchmark(variant="baseline", root=tmp_path)
    assert backend.batch_calls == 1
    assert resumed["generation_cache"]["generated"] == 0
    assert resumed["generation_cache"]["cached"] == 2


def test_batched_cache_prefill_orders_only_by_prompt_length(tmp_path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reference},
            ],
            "meta": {"content_hash": f"hash-{index}"},
        }
        for index, (question, reference) in enumerate(
            [
                ("four", "x"),
                ("a", "reference intentionally longest here"),
                ("bbb", "yy"),
                ("cc", "zzz"),
            ]
        )
    ]

    class OrderedBackend:
        def __init__(self) -> None:
            self.questions = []

        def generate_batch(self, messages_batch, generation):
            self.questions.extend(messages[-1]["content"] for messages in messages_batch)
            return [
                Generation("ok", 1.0, 1, 1, batch_size=2, batch_completion_tokens_total=2)
                for _ in messages_batch
            ]

    backend = OrderedBackend()
    generated = benchmark_pipeline._prefill_generation_cache(
        backend=backend,
        input_records=records,
        variant="baseline",
        backend_config={"backend": "fake"},
        generation={
            "seed": 3407,
            "batch_size": 2,
            "batch_order": "prompt_length_ascending",
            "max_new_tokens": 8,
        },
        cache_dir=tmp_path / "cache",
    )

    assert backend.questions == ["a", "cc", "bbb", "four"]
    assert generated == {"hash-0", "hash-1", "hash-2", "hash-3"}

    # Simulate power loss after only one member of the first deterministic
    # group survived. The whole affected group is regenerated, while the
    # complete second group remains cached and keeps its membership.
    (tmp_path / "cache" / "hash-1.json").unlink()
    regenerated = benchmark_pipeline._prefill_generation_cache(
        backend=backend,
        input_records=records,
        variant="baseline",
        backend_config={"backend": "fake"},
        generation={
            "seed": 3407,
            "batch_size": 2,
            "batch_order": "prompt_length_ascending",
            "max_new_tokens": 8,
        },
        cache_dir=tmp_path / "cache",
    )

    assert backend.questions == ["a", "cc", "bbb", "four", "a", "cc"]
    assert regenerated == {"hash-1", "hash-3"}

    corrupted_path = tmp_path / "cache" / "hash-2.json"
    corrupted = json.loads(corrupted_path.read_text(encoding="utf-8"))
    corrupted["generation"]["batch_group_position"] = 99
    corrupted_path.write_text(json.dumps(corrupted), encoding="utf-8")
    repaired = benchmark_pipeline._prefill_generation_cache(
        backend=backend,
        input_records=records,
        variant="baseline",
        backend_config={"backend": "fake"},
        generation={
            "seed": 3407,
            "batch_size": 2,
            "batch_order": "prompt_length_ascending",
            "max_new_tokens": 8,
        },
        cache_dir=tmp_path / "cache",
    )

    assert backend.questions[-2:] == ["bbb", "four"]
    assert repaired == {"hash-0", "hash-2"}


def test_hip_runtime_metadata_records_gfx_architecture() -> None:
    class Properties:
        name = "AMD Radeon Graphics"
        total_memory = 32624 * 1024**2
        major = 12
        minor = 0
        gcnArchName = "gfx1201"

    class Cuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return Properties()

    class Version:
        cuda = None
        hip = "7.2.53211"

    class Torch:
        __version__ = "2.12.1+rocm7.2"
        cuda = Cuda()
        version = Version()

    runtime = _cuda_runtime_metadata(Torch())
    assert runtime["accelerator"] == "rocm"
    assert runtime["gcn_arch"] == "gfx1201"
    assert runtime["torch_hip_version"] == "7.2.53211"
    assert "compute_capability" not in runtime


def test_report_runtime_signature_supports_rocm() -> None:
    result = {
        "runtime": {
            "backend": "unsloth",
            "device": "AMD Radeon Graphics",
            "torch_version": "2.12.1+rocm7.2",
            "accelerator": "rocm",
            "gcn_arch": "gfx1201",
            "torch_hip_version": "7.2.53211",
        }
    }

    signature = benchmark_report._in_process_runtime_signature(result, "baseline")

    assert "rocm" in signature


def test_report_runtime_signature_supports_legacy_cuda() -> None:
    result = _result(0.8)
    del result["runtime"]["accelerator"]
    signature = benchmark_report._in_process_runtime_signature(result, "baseline")
    assert "cuda" in signature
    assert "12.0" in signature
