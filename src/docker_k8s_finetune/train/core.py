from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import PipelineError
from ..schema import ChatRecord


@dataclass(frozen=True)
class TrainingAttempt:
    max_seq_length: int
    batch_size: int
    gradient_accumulation_steps: int
    output_dir: Path
    max_steps: int = -1
    smoke_test: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def resolve_model_reference(config: Mapping[str, Any], root: Path) -> str:
    model = config["model"]
    local = root / str(model["local_path"])
    required = ("config.json", "tokenizer_config.json", "model.safetensors.index.json")
    if model.get("prefer_local_if_present") and all((local / name).is_file() for name in required):
        if not list(local.glob("*.safetensors")):
            raise PipelineError(f"Local model has no safetensors shards: {local}")
        return str(local.resolve())
    return str(model["name"])


def inspect_local_model(model_path: Path, *, instruction_delimiter: str, response_delimiter: str) -> dict[str, Any]:
    if not model_path.is_dir():
        raise PipelineError(f"Local model directory does not exist: {model_path}")
    try:
        model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text(encoding="utf-8"))
        index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot inspect local model {model_path}: {exc}") from exc
    architecture = model_config.get("architectures", [None])[0]
    if architecture != "Qwen3_5ForConditionalGeneration":
        raise PipelineError(f"Unexpected local model architecture: {architecture}")
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing = [name for name in shards if not (model_path / name).is_file()]
    if not shards or missing:
        raise PipelineError(f"Local model is missing weight shards: {missing or 'no shards indexed'}")
    template = str(tokenizer_config.get("chat_template", ""))
    for delimiter in (instruction_delimiter, response_delimiter):
        if delimiter not in template.replace(" + message.role + ", "user") and delimiter not in template:
            # Qwen's Jinja template builds role delimiters dynamically; the rendered
            # literal is validated again in the GPU smoke test.
            if "<|im_start|>" not in template:
                raise PipelineError("Local chat template is incompatible with response-only masking")
    return {
        "architecture": architecture,
        "model_type": model_config.get("model_type"),
        "shards": len(shards),
        "weights_bytes": sum((model_path / name).stat().st_size for name in shards),
        "chat_template_present": bool(template),
    }


def load_chatml(path: Path, *, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineError(f"Training dataset does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        eligible_index = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if eligible_index < offset:
                eligible_index += 1
                continue
            try:
                value = json.loads(line)
                ChatRecord(messages=value["messages"], meta=value["meta"]).validate()
            except (KeyError, TypeError, json.JSONDecodeError, PipelineError) as exc:
                raise PipelineError(f"Invalid ChatML in {path}:{line_number}: {exc}") from exc
            records.append(value)
            eligible_index += 1
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise PipelineError(f"Training dataset is empty: {path}")
    return records


def format_chatml(records: Iterable[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    formatted = []
    for record in records:
        text = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not text.strip():
            raise PipelineError("Tokenizer produced an empty training example")
        formatted.append({"text": text, "meta": dict(record["meta"])})
    return formatted


def latest_checkpoint(output_dir: Path) -> str | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            if path.is_dir():
                try:
                    candidates.append((int(path.name.rsplit("-", 1)[1]), path))
                except ValueError:
                    continue
    return str(max(candidates)[1]) if candidates else None


def build_attempt(config: Mapping[str, Any], root: Path, *, smoke_test: bool, fallback: bool = False) -> TrainingAttempt:
    trainer = config["trainer"]
    if smoke_test:
        smoke = config["smoke_test"]
        return TrainingAttempt(
            max_seq_length=int(smoke["max_seq_length"]),
            batch_size=int(smoke["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(smoke["gradient_accumulation_steps"]),
            output_dir=root / str(smoke["output_dir"]),
            max_steps=int(smoke["max_steps"]),
            smoke_test=True,
        )
    if fallback:
        fallback_config = trainer["oom_fallback"]
        return TrainingAttempt(
            max_seq_length=int(config["model"]["fallback_max_seq_length_on_oom"]),
            batch_size=int(fallback_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(fallback_config["gradient_accumulation_steps"]),
            output_dir=root / str(config["output"]["checkpoints"]),
        )
    return TrainingAttempt(
        max_seq_length=int(config["model"]["max_seq_length"]),
        batch_size=int(trainer["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(trainer["gradient_accumulation_steps"]),
        output_dir=root / str(config["output"]["checkpoints"]),
    )


def validate_training_config(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    architecture = config.get("architecture", {})
    lora = config.get("lora", {})
    trainer = config.get("trainer", {})
    if model.get("loader") != "FastVisionModel" or model.get("architecture") != "qwen3_5_vlm_text_only":
        raise PipelineError("Qwen3.5-4B must use FastVisionModel in text-only mode")
    if architecture.get("text_only") is not True or architecture.get("finetune_vision_layers") is not False:
        raise PipelineError("Text-only training must keep vision layers frozen")
    if lora.get("target_modules") != "auto":
        raise PipelineError("Qwen3.5 hybrid attention requires Unsloth automatic module selection")
    if trainer.get("response_only_loss") is not True:
        raise PipelineError("Response-only loss is required")
    if int(trainer.get("effective_batch_size", -1)) != int(trainer["per_device_train_batch_size"]) * int(trainer["gradient_accumulation_steps"]):
        raise PipelineError("Configured effective_batch_size is inconsistent")
