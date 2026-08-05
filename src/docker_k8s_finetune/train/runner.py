from __future__ import annotations

import gc
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..errors import PipelineError
from ..io import file_sha256
from ..schema import utc_now_iso
from .callbacks import build_vram_callback
from .core import (
    TrainingAttempt,
    build_attempt,
    format_chatml,
    latest_checkpoint,
    load_chatml,
    inspect_local_model,
    resolve_model_reference,
    validate_training_config,
)

LOGGER = logging.getLogger(__name__)


def _warmup_steps(
    *, examples: int, epochs: float, batch_size: int, gradient_accumulation_steps: int,
    max_steps: int, warmup_ratio: float,
) -> int:
    if warmup_ratio <= 0:
        return 0
    if max_steps > 0:
        total_steps = max_steps
    else:
        batches_per_epoch = math.ceil(examples / batch_size)
        updates_per_epoch = max(1, math.ceil(batches_per_epoch / gradient_accumulation_steps))
        total_steps = max(1, math.ceil(updates_per_epoch * epochs))
    return max(1, math.ceil(total_steps * warmup_ratio))


def _verify_final_split(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    data = config["data"]
    statistics_path = root / str(data["split_statistics"])
    if not statistics_path.is_file():
        raise PipelineError(f"Final split statistics do not exist: {statistics_path}")
    try:
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read final split statistics: {exc}") from exc
    if statistics.get("provisional"):
        raise PipelineError("Refusing full training on provisional dataset splits")
    for split_name, data_key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        path = root / str(data[data_key])
        expected = statistics.get("outputs", {}).get(split_name, {}).get("sha256")
        if not path.is_file() or not expected:
            raise PipelineError(f"Final {split_name} split or its recorded hash is missing")
        if file_sha256(path) != expected:
            raise PipelineError(f"Final {split_name} split hash differs from split_statistics.json")
    return statistics


def run_training_preflight(
    *, config_path: Path = Path("config/training.yaml"), smoke_test: bool = True,
) -> dict[str, Any]:
    root = Path.cwd()
    config = load_yaml(config_path)
    validate_training_config(config)
    model_ref = resolve_model_reference(config, root)
    model_path = Path(model_ref)
    trainer = config["trainer"]
    model_report = inspect_local_model(
        model_path,
        instruction_delimiter=str(trainer["instruction_delimiter"]),
        response_delimiter=str(trainer["response_delimiter"]),
    ) if model_path.is_dir() else {"remote_reference": model_ref}
    if smoke_test:
        smoke = config["smoke_test"]
        input_path = root / str(smoke["input"])
        train_records = load_chatml(input_path, limit=int(smoke["train_records"]))
        validation_records = load_chatml(
            input_path,
            limit=int(smoke["validation_records"]),
            offset=int(smoke["train_records"]),
        )
    else:
        data = config["data"]
        _verify_final_split(config, root)
        train_records = load_chatml(root / str(data["train"]))
        validation_records = load_chatml(root / str(data["validation"]))
        input_path = root / str(data["train"])
    result = {
        "status": "ready",
        "mode": "smoke" if smoke_test else "full",
        "model_reference": model_ref,
        "model": model_report,
        "dataset": {
            "input": str(input_path),
            "train_records_checked": len(train_records),
            "validation_records_checked": len(validation_records),
        },
    }
    return result


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _dtype(value: str, torch: Any) -> Any:
    supported = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": None}
    if value not in supported:
        raise PipelineError(f"Unsupported training dtype: {value}")
    return supported[value]


def _summarize_vram(path: Path, run_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("run_id") == run_id:
            values.append(value)
    if not values:
        return None
    return {
        "log": str(path),
        "epochs_recorded": len(values),
        "peak_allocated_gib": max(float(value["peak_allocated_gib"]) for value in values),
        "peak_reserved_gib": max(float(value["peak_reserved_gib"]) for value in values),
        "maximum_device_used_gib": max(float(value["device_used_gib"]) for value in values),
        "minimum_device_free_gib": min(float(value["device_free_gib"]) for value in values),
        "device_total_gib": float(values[-1]["device_total_gib"]),
        "device": values[-1]["device"],
    }


def _train_once(config: Mapping[str, Any], root: Path, attempt: TrainingAttempt, *, export: bool) -> dict[str, Any]:
    try:
        import torch
        # Unsloth must patch Transformers/TRL before either library is imported.
        from unsloth import FastVisionModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth.chat_templates import train_on_responses_only
    except ImportError as exc:
        raise PipelineError("Training dependencies are unavailable; run through compose.train.yaml") from exc

    if not torch.cuda.is_available():
        raise PipelineError("CUDA GPU is required for QLoRA training")

    smoke = config["smoke_test"]
    train_limit = int(smoke["train_records"]) if attempt.smoke_test else None
    validation_limit = int(smoke["validation_records"]) if attempt.smoke_test else None
    if attempt.smoke_test:
        smoke_input = root / str(smoke["input"])
        train_records = load_chatml(smoke_input, limit=train_limit)
        validation_records = load_chatml(smoke_input, limit=validation_limit, offset=int(smoke["train_records"]))
    else:
        data_config = config["data"]
        _verify_final_split(config, root)
        train_records = load_chatml(root / str(data_config["train"]))
        validation_records = load_chatml(root / str(data_config["validation"]))
    model_ref = resolve_model_reference(config, root)
    LOGGER.info("Loading %s with FastVisionModel (text only, seq=%d)", model_ref, attempt.max_seq_length)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_ref,
        max_seq_length=attempt.max_seq_length,
        dtype=_dtype(str(config["model"]["dtype"]), torch),
        load_in_4bit=bool(config["model"]["load_in_4bit"]),
    )
    lora = config["lora"]
    architecture = config["architecture"]
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(architecture["finetune_vision_layers"]),
        finetune_language_layers=bool(architecture["finetune_language_layers"]),
        finetune_attention_modules=bool(architecture["finetune_attention_modules"]),
        finetune_mlp_modules=bool(architecture["finetune_mlp_modules"]),
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        use_gradient_checkpointing=lora["use_gradient_checkpointing"],
        random_state=int(config["seed"]),
    )
    FastVisionModel.for_training(model)
    train_dataset = Dataset.from_list(format_chatml(train_records, tokenizer))
    validation_dataset = Dataset.from_list(format_chatml(validation_records, tokenizer))
    trainer_config = config["trainer"]
    metrics_root = root / str(config["output"]["metrics"])
    metrics_root.mkdir(parents=True, exist_ok=True)
    checkpoint = None if attempt.smoke_test else latest_checkpoint(attempt.output_dir)
    run_id = utc_now_iso()
    vram_path = metrics_root / ("vram_smoke.jsonl" if attempt.smoke_test else "vram_by_epoch.jsonl")
    tensorboard_path = metrics_root / "tensorboard"
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_path)
    sft_args = SFTConfig(
        output_dir=str(attempt.output_dir),
        num_train_epochs=float(trainer_config["epochs"]),
        max_steps=attempt.max_steps,
        per_device_train_batch_size=attempt.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=attempt.gradient_accumulation_steps,
        learning_rate=float(trainer_config["learning_rate"]),
        lr_scheduler_type=str(trainer_config["scheduler"]),
        warmup_steps=_warmup_steps(
            examples=len(train_dataset),
            epochs=float(trainer_config["epochs"]),
            batch_size=attempt.batch_size,
            gradient_accumulation_steps=attempt.gradient_accumulation_steps,
            max_steps=attempt.max_steps,
            warmup_ratio=float(trainer_config["warmup_ratio"]),
        ),
        weight_decay=float(trainer_config["weight_decay"]),
        optim=str(trainer_config["optimizer"]),
        eval_strategy=str(trainer_config["eval_strategy"]),
        eval_steps=int(trainer_config["eval_steps"]),
        logging_steps=1 if attempt.smoke_test else int(trainer_config["logging_steps"]),
        save_steps=int(trainer_config["save_steps"]),
        save_total_limit=int(trainer_config["save_total_limit"]),
        report_to=str(trainer_config["report_to"]),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        bf16=str(config["model"]["dtype"]) == "bfloat16",
        fp16=str(config["model"]["dtype"]) == "float16",
        max_length=attempt.max_seq_length,
        dataset_text_field="text",
        packing=bool(trainer_config["packing"]),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=sft_args,
        callbacks=[build_vram_callback(vram_path, run_id=run_id, append=checkpoint is not None)],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=str(trainer_config["instruction_delimiter"]),
        response_part=str(trainer_config["response_delimiter"]),
    )
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    evaluation_metrics = trainer.evaluate()

    result = {
        "status": "trained",
        "smoke_test": attempt.smoke_test,
        "model_reference": model_ref,
        "max_seq_length": attempt.max_seq_length,
        "batch_size": attempt.batch_size,
        "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
        "effective_batch_size": attempt.effective_batch_size,
        "resumed_from": checkpoint,
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
        "vram": _summarize_vram(vram_path, run_id),
    }
    if not export:
        return result

    output = config["output"]
    adapter_dir = root / str(output["adapter"])
    merged_dir = root / str(output["merged"])
    gguf_dir = root / str(output["gguf"])
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method=list(output["gguf_quantizations"]),
    )
    result["exports"] = {
        "adapter": str(adapter_dir),
        "merged": str(merged_dir),
        "gguf": str(gguf_dir),
        "quantizations": list(output["gguf_quantizations"]),
    }
    return result


def run_training(
    *, config_path: Path = Path("config/training.yaml"), smoke_test: bool = False,
    export: bool = True, preflight_only: bool = False,
) -> dict[str, Any]:
    if preflight_only:
        return run_training_preflight(config_path=config_path, smoke_test=smoke_test)
    root = Path.cwd()
    config = load_yaml(config_path)
    validate_training_config(config)
    attempt = build_attempt(config, root, smoke_test=smoke_test)
    try:
        result = _train_once(config, root, attempt, export=export and not smoke_test)
    except RuntimeError as exc:
        fallback_enabled = bool(config["trainer"]["oom_fallback"]["enabled"])
        if smoke_test or not fallback_enabled or not _is_cuda_oom(exc):
            raise
        LOGGER.warning("CUDA OOM with primary settings; retrying the configured conservative profile")
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        fallback = build_attempt(config, root, smoke_test=False, fallback=True)
        result = _train_once(config, root, fallback, export=export)
        result["oom_fallback_used"] = True
    report_path = (root / str(config["output"]["metrics"])) / ("smoke_result.json" if smoke_test else "training_result.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result
