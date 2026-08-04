from __future__ import annotations

import json
from pathlib import Path

import pytest

from docker_k8s_finetune.errors import PipelineError
from docker_k8s_finetune.train.core import (
    build_attempt,
    format_chatml,
    latest_checkpoint,
    load_chatml,
    resolve_model_reference,
    inspect_local_model,
    validate_training_config,
)


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
        "model": {"loader": "FastVisionModel", "architecture": "qwen3_5_vlm_text_only", "local_path": "model", "name": "remote", "prefer_local_if_present": True, "max_seq_length": 8192, "fallback_max_seq_length_on_oom": 4096},
        "architecture": {"text_only": True, "finetune_vision_layers": False},
        "lora": {"target_modules": "auto"},
        "trainer": {"response_only_loss": True, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4, "effective_batch_size": 16, "oom_fallback": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16}},
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


def test_resolve_model_requires_complete_local_weights(tmp_path: Path) -> None:
    config = _config()
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json", "model-1.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    assert resolve_model_reference(config, tmp_path) == str(model.resolve())


def test_attempts_and_latest_checkpoint_are_deterministic(tmp_path: Path) -> None:
    config = _config()
    assert build_attempt(config, tmp_path, smoke_test=False).effective_batch_size == 16
    assert build_attempt(config, tmp_path, smoke_test=False, fallback=True).max_seq_length == 4096
    output = tmp_path / "artifacts" / "checkpoints"
    (output / "checkpoint-9").mkdir(parents=True)
    (output / "checkpoint-100").mkdir()
    assert latest_checkpoint(output).endswith("checkpoint-100")


def test_training_config_rejects_manual_targets() -> None:
    config = _config()
    validate_training_config(config)
    config["lora"]["target_modules"] = ["q_proj"]
    with pytest.raises(PipelineError, match="hybrid attention"):
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
