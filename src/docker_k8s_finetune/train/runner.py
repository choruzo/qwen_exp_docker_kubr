from __future__ import annotations

import gc
from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from ..benchmark.core import judge_provenance, syntax_provenance
from ..config import load_yaml
from ..errors import PipelineError
from ..dedupe.exact import canonical_content
from ..io import atomic_write_json, content_hash, file_sha256, stable_json
from ..schema import utc_now_iso
from .callbacks import build_checkpoint_provenance_callback, build_vram_callback
from .core import (
    TrainingAttempt,
    artifact_files_manifest,
    build_attempt,
    finalize_gguf_export,
    filter_formatted_by_token_length,
    format_chatml,
    latest_checkpoint,
    load_chatml,
    inspect_local_model,
    resolve_model_reference,
    resolve_text_tokenizer,
    verify_local_model_provenance,
    validate_training_config,
)

LOGGER = logging.getLogger(__name__)


def _rocm_math_library_provenance(*, required: bool = False) -> dict[str, Any]:
    knobs = (
        "PYTORCH_HIP_ALLOC_CONF",
        "TORCH_BLAS_PREFER_HIPBLASLT",
        "ROCBLAS_USE_HIPBLASLT",
        "DISABLE_ADDMM_HIP_LT",
    )
    override = os.environ.get("HIPBLASLT_TENSILE_LIBPATH")
    provenance: dict[str, Any] = {
        "hipblaslt_tensile_libpath": override,
        "environment": {name: os.environ[name] for name in knobs if name in os.environ},
    }
    if not override:
        if required:
            raise PipelineError("This ROCm profile requires a patched hipBLASLt overlay")
        return provenance
    library = Path(override)
    if not library.is_dir():
        raise PipelineError(f"HIPBLASLT_TENSILE_LIBPATH is not a directory: {library}")
    objects = sorted(library.glob("TensileLibrary_BB_BB*gfx1201.co"))
    if len(objects) != 4:
        raise PipelineError("Expected four patched BF16 gfx1201 hipBLASLt objects")
    files: dict[str, str] = {}
    for obj in objects:
        for path in (obj, obj.with_suffix(".dat")):
            if not path.is_file() or path.is_symlink():
                raise PipelineError(f"Patched hipBLASLt file is missing or points to stock: {path}")
            files[path.name] = file_sha256(path)
    provenance["patched_files_sha256"] = files
    return provenance


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


def _verify_validation_redaction(
    config: Mapping[str, Any], statistics: Mapping[str, Any], path: Path,
) -> dict[str, Any]:
    """Accept the one pinned post-split credential redaction without changing the manifest."""
    if config.get("runtime", {}).get("accelerator") != "rocm":
        raise PipelineError("Validation redaction exception is restricted to the ROCm profile")
    exception = config.get("data", {}).get("validation_redaction")
    if not isinstance(exception, Mapping):
        raise PipelineError("Validation split differs from the frozen manifest without a redaction exception")
    required = {
        "manifest_sha256", "manifest_bytes", "actual_sha256", "actual_bytes",
        "record_line", "source_record_id", "recorded_content_hash", "redacted_content_hash",
    }
    if not required.issubset(exception):
        raise PipelineError("Validation redaction exception is incomplete")
    recorded = statistics.get("outputs", {}).get("validation", {})
    actual_hash = file_sha256(path)
    expected_values = (
        (recorded.get("sha256"), exception.get("manifest_sha256")),
        (recorded.get("bytes"), exception.get("manifest_bytes")),
        (actual_hash, exception.get("actual_sha256")),
        (path.stat().st_size, exception.get("actual_bytes")),
    )
    if any(observed != expected for observed, expected in expected_values):
        raise PipelineError("Validation redaction exception does not match the pinned files")

    mismatches: list[tuple[int, str, str, str]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                meta = record["meta"]
                recorded_hash = str(meta["content_hash"])
                actual_content_hash = content_hash(canonical_content(record))
                if actual_content_hash != recorded_hash:
                    mismatches.append((
                        line_number, str(meta["source_record_id"]),
                        recorded_hash, actual_content_hash,
                    ))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PipelineError("Cannot audit validation record hashes") from exc
    expected_mismatch = (
        int(exception["record_line"]), str(exception["source_record_id"]),
        str(exception["recorded_content_hash"]), str(exception["redacted_content_hash"]),
    )
    if mismatches != [expected_mismatch]:
        raise PipelineError("Validation record hashes do not match the pinned redaction exception")

    reconciled = deepcopy(dict(statistics))
    reconciled["outputs"]["validation"] = {
        **recorded,
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
        "manifest_sha256": exception["manifest_sha256"],
        "redaction_exception": {key: exception[key] for key in (
            "record_line", "source_record_id", "recorded_content_hash", "redacted_content_hash",
        )},
    }
    return reconciled


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
            if split_name == "validation":
                statistics = _verify_validation_redaction(config, statistics, path)
            else:
                raise PipelineError(f"Final {split_name} split hash differs from split_statistics.json")
    return statistics


def _verify_baseline(
    config: Mapping[str, Any], root: Path, split_statistics: Mapping[str, Any],
) -> dict[str, Any] | None:
    prerequisite = config.get("prerequisites", {})
    if not prerequisite.get("require_final_baseline", False):
        return None
    path = root / str(prerequisite["baseline_results"])
    if not path.is_file():
        raise PipelineError(f"Final baseline must be completed before full training: {path}")
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read baseline prerequisite: {exc}") from exc
    if baseline.get("variant") != "baseline" or baseline.get("provisional"):
        raise PipelineError("Full training requires a non-provisional baseline result")
    expected_hash = split_statistics.get("outputs", {}).get("test", {}).get("sha256")
    if not expected_hash or baseline.get("split_test_sha256") != expected_hash:
        raise PipelineError("Baseline result does not match the frozen test split")
    model_config = config.get("model")
    if isinstance(model_config, Mapping):
        model_reference = resolve_model_reference(config, root)
        model_path = Path(model_reference)
        expected_model_provenance = (
            verify_local_model_provenance(model_path, model_config)
            if model_path.is_dir()
            else {"repository": model_reference, "revision": model_config.get("revision")}
        )
        if baseline.get("model_provenance") != expected_model_provenance:
            raise PipelineError("Baseline result uses different base-model provenance")
    completion = baseline.get("completion") or {}
    if not all(completion.get(stage, False) for stage in ("full_test", "generation", "syntax", "judge")):
        raise PipelineError("Baseline result has not completed generation, syntax and judge stages")
    benchmark_config_path = root / str(prerequisite["benchmark_config"])
    if not benchmark_config_path.is_file():
        raise PipelineError(f"Benchmark configuration is missing: {benchmark_config_path}")
    benchmark_config = load_yaml(benchmark_config_path)
    if baseline.get("judge_provenance") != judge_provenance(benchmark_config, root):
        raise PipelineError("Baseline judge prompt or manifest differs from the current benchmark configuration")
    if benchmark_config.get("semantic_similarity") and baseline.get(
        "semantic_similarity_config"
    ) != benchmark_config["semantic_similarity"]:
        raise PipelineError("Baseline semantic similarity model differs from benchmark configuration")
    if benchmark_config.get("syntax_validation", {}).get("enabled") and baseline.get(
        "syntax_provenance"
    ) != syntax_provenance(benchmark_config, root):
        raise PipelineError("Baseline syntax validator configuration differs from current configuration")
    expected_judge = benchmark_config.get("llm_judge", {}).get("model_identity")
    if expected_judge:
        judge_runtime = baseline.get("judge_runtime")
        if not isinstance(judge_runtime, Mapping) or judge_runtime.get("expected") != expected_judge:
            raise PipelineError("Baseline judge model identity differs from the benchmark configuration")
        served_judge = judge_runtime.get("served")
        if not isinstance(served_judge, Mapping) or any(
            served_judge.get(field) != expected_judge.get(field)
            for field in ("alias", "filename", "quantization")
        ):
            raise PipelineError("Baseline does not prove the served judge model identity")
    return baseline


def run_training_preflight(
    *, config_path: Path = Path("config/training.yaml"), smoke_test: bool = True,
) -> dict[str, Any]:
    root = Path.cwd()
    config = load_yaml(config_path)
    validate_training_config(config)
    model_ref = resolve_model_reference(config, root)
    model_path = Path(model_ref)
    trainer = config["trainer"]
    if model_path.is_dir():
        provenance = verify_local_model_provenance(model_path, config["model"])
        model_report = inspect_local_model(
            model_path,
            instruction_delimiter=str(trainer["instruction_delimiter"]),
            response_delimiter=str(trainer["response_delimiter"]),
        )
        model_report["provenance"] = provenance
    else:
        model_report = {
            "remote_reference": model_ref,
            "revision": str(config["model"]["revision"]),
        }
    if smoke_test:
        smoke = config["smoke_test"]
        input_path = root / str(smoke["input"])
        train_records = load_chatml(
            input_path, limit=int(smoke["train_records"]), offset=int(smoke.get("train_offset", 0))
        )
        validation_input = root / str(smoke.get("validation_input", smoke["input"]))
        validation_records = load_chatml(
            validation_input,
            limit=int(smoke["validation_records"]),
            offset=int(smoke.get("validation_offset", 0)) if "validation_input" in smoke
            else int(smoke["train_records"]),
        )
    else:
        data = config["data"]
        statistics = _verify_final_split(config, root)
        _verify_baseline(config, root, statistics)
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
            **({"validation_identity": statistics["outputs"]["validation"]} if not smoke_test else {}),
        },
    }
    return result


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _oom_state_contract(config: Mapping[str, Any], root: Path) -> str:
    statistics_path = root / str(config.get("data", {}).get("split_statistics", ""))
    statistics_sha256 = file_sha256(statistics_path) if statistics_path.is_file() else None
    return content_hash(stable_json({
        "training_config": config,
        "split_statistics_sha256": statistics_sha256,
    }))


def _load_oom_state(path: Path, *, contract: str) -> list[str]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if value.get("contract") != contract or not isinstance(value.get("failed_profiles"), list):
        return []
    return [str(profile) for profile in value["failed_profiles"]]


def _write_oom_state(path: Path, *, contract: str, failed_profiles: list[str]) -> None:
    atomic_write_json(path, {
        "version": 1,
        "updated_at": utc_now_iso(),
        "contract": contract,
        "failed_profiles": list(dict.fromkeys(failed_profiles)),
    })


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


def _train_once(
    config: Mapping[str, Any], root: Path, attempt: TrainingAttempt,
    *, export: bool, resume_from_latest: bool = False,
) -> dict[str, Any]:
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
        raise PipelineError("PyTorch cannot see a GPU for training")
    accelerator = config.get("runtime", {}).get("accelerator", "cuda")
    if accelerator == "rocm":
        if not torch.version.hip:
            raise PipelineError("ROCm profile requires a PyTorch HIP build")
        expected_gfx = str(config["runtime"].get("expected_gfx", ""))
        properties = torch.cuda.get_device_properties(0)
        observed_gfx = str(getattr(properties, "gcnArchName", ""))
        if expected_gfx and not observed_gfx.startswith(expected_gfx):
            raise PipelineError(f"Expected {expected_gfx}; found {observed_gfx}")
        if int(properties.total_memory) < 30 * 1024 ** 3:
            raise PipelineError("ROCm profile requires the 32 GB discrete GPU")

    rocm_math_library = (
        _rocm_math_library_provenance(
            required=bool(config.get("runtime", {}).get("require_hipblaslt_override", False))
        )
        if accelerator == "rocm" else None
    )

    smoke = config["smoke_test"]
    train_limit = int(smoke["train_records"]) if attempt.smoke_test else None
    validation_limit = int(smoke["validation_records"]) if attempt.smoke_test else None
    split_statistics = None
    if attempt.smoke_test:
        smoke_input = root / str(smoke["input"])
        train_records = load_chatml(
            smoke_input, limit=train_limit, offset=int(smoke.get("train_offset", 0))
        )
        validation_input = root / str(smoke.get("validation_input", smoke["input"]))
        validation_records = load_chatml(
            validation_input, limit=validation_limit,
            offset=int(smoke.get("validation_offset", 0)) if "validation_input" in smoke
            else int(smoke["train_records"]),
        )
    else:
        data_config = config["data"]
        split_statistics = _verify_final_split(config, root)
        _verify_baseline(config, root, split_statistics)
        train_records = load_chatml(root / str(data_config["train"]))
        validation_records = load_chatml(root / str(data_config["validation"]))
    model_ref = resolve_model_reference(config, root)
    base_provenance = None
    if Path(model_ref).is_dir():
        base_provenance = verify_local_model_provenance(Path(model_ref), config["model"])
    LOGGER.info("Loading %s with FastVisionModel (text only, seq=%d)", model_ref, attempt.max_seq_length)
    model_load_kwargs = {
        "model_name": model_ref,
        "max_seq_length": attempt.max_seq_length,
        "dtype": _dtype(str(config["model"]["dtype"]), torch),
        "load_in_4bit": bool(config["model"]["load_in_4bit"]),
    }
    if not Path(model_ref).is_dir():
        model_load_kwargs["revision"] = str(config["model"]["revision"])
    model, processor = FastVisionModel.from_pretrained(
        **model_load_kwargs,
    )
    tokenizer = resolve_text_tokenizer(processor)
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
    trainer_config = config["trainer"]
    formatted_train, train_length_filter = filter_formatted_by_token_length(
        format_chatml(train_records, tokenizer),
        tokenizer,
        max_length=attempt.max_seq_length,
        batch_size=int(trainer_config["length_audit_batch_size"]),
    )
    formatted_validation, validation_length_filter = filter_formatted_by_token_length(
        format_chatml(validation_records, tokenizer),
        tokenizer,
        max_length=attempt.max_seq_length,
        batch_size=int(trainer_config["length_audit_batch_size"]),
    )
    base_identity = base_provenance or {
        "repository": config["model"]["name"],
        "revision": config["model"]["revision"],
    }
    training_split_contract = (
        {
            name: split_statistics["outputs"][name]["sha256"]
            for name in ("train", "validation", "test")
        }
        if split_statistics is not None
        else {"smoke_input": file_sha256(smoke_input), "smoke_validation": file_sha256(validation_input)}
    )
    checkpoint_contract = {
        "version": 1,
        "config_sha256": content_hash(stable_json(config)),
        "base_model": base_identity,
        "training_split": training_split_contract,
        "validation_identity": (
            split_statistics["outputs"]["validation"] if split_statistics is not None else None
        ),
        "training_profile": {
            "name": attempt.profile,
            "max_seq_length": attempt.max_seq_length,
            "batch_size": attempt.batch_size,
            "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
            "eval_batch_size": 1 if attempt.smoke_test else int(trainer_config["per_device_eval_batch_size"]),
            "eval_steps": int(smoke.get("eval_steps", 1)) if attempt.smoke_test else int(trainer_config["eval_steps"]),
            "packing": bool(trainer_config["packing"]),
            "train_sampling_strategy": str(trainer_config["train_sampling_strategy"]),
        },
        "processing": {
            "processor_class": type(processor).__name__,
            "text_tokenizer_class": type(tokenizer).__name__,
        },
        "length_filter": {
            "train": train_length_filter,
            "validation": validation_length_filter,
        },
    }
    if rocm_math_library is not None:
        checkpoint_contract["rocm_math_library"] = rocm_math_library
    checkpoint_fingerprint = content_hash(stable_json(checkpoint_contract))
    train_dataset = Dataset.from_list(formatted_train)
    validation_dataset = Dataset.from_list(formatted_validation)
    metrics_root = root / str(config["output"]["metrics"])
    metrics_root.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if resume_from_latest or (not attempt.smoke_test and trainer_config["resume_from_checkpoint"] == "auto"):
        checkpoint = latest_checkpoint(attempt.output_dir, expected_fingerprint=checkpoint_fingerprint)
        if resume_from_latest and checkpoint is None:
            raise PipelineError("No complete checkpoint matches this ROCm training contract")
    run_id = utc_now_iso()
    vram_path = metrics_root / (
        str(smoke.get("vram_name", "vram_smoke.jsonl")) if attempt.smoke_test else "vram_by_epoch.jsonl"
    )
    tensorboard_path = metrics_root / "tensorboard"
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_path)
    sft_args = SFTConfig(
        output_dir=str(attempt.output_dir),
        num_train_epochs=float(trainer_config["epochs"]),
        max_steps=attempt.max_steps,
        per_device_train_batch_size=attempt.batch_size,
        per_device_eval_batch_size=(
            1 if attempt.smoke_test else int(trainer_config["per_device_eval_batch_size"])
        ),
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
        eval_steps=int(smoke.get("eval_steps", 1)) if attempt.smoke_test else int(trainer_config["eval_steps"]),
        logging_steps=1 if attempt.smoke_test else int(trainer_config["logging_steps"]),
        save_steps=int(smoke.get("save_steps", 1)) if attempt.smoke_test else int(trainer_config["save_steps"]),
        save_total_limit=int(trainer_config["save_total_limit"]),
        report_to=str(trainer_config["report_to"]),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        bf16=str(config["model"]["dtype"]) == "bfloat16",
        fp16=str(config["model"]["dtype"]) == "float16",
        max_length=attempt.max_seq_length,
        dataset_text_field="text",
        packing=bool(trainer_config["packing"]),
        train_sampling_strategy=str(trainer_config["train_sampling_strategy"]),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=sft_args,
        callbacks=[
            build_vram_callback(vram_path, run_id=run_id, append=checkpoint is not None),
            build_checkpoint_provenance_callback(
                checkpoint_contract, fingerprint=checkpoint_fingerprint
            ),
        ],
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
        "training_profile": attempt.profile,
        "model_reference": model_ref,
        "runtime": {
            "accelerator": accelerator,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gcn_arch": str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            **({"rocm_math_library": rocm_math_library} if rocm_math_library is not None else {}),
        },
        "max_seq_length": attempt.max_seq_length,
        "batch_size": attempt.batch_size,
        "gradient_accumulation_steps": attempt.gradient_accumulation_steps,
        "effective_batch_size": attempt.effective_batch_size,
        "eval_batch_size": 1 if attempt.smoke_test else int(trainer_config["per_device_eval_batch_size"]),
        "eval_steps": int(smoke.get("eval_steps", 1)) if attempt.smoke_test else int(trainer_config["eval_steps"]),
        "packing": bool(trainer_config["packing"]),
        "train_sampling_strategy": str(trainer_config["train_sampling_strategy"]),
        "resumed_from": checkpoint,
        "metrics": train_result.metrics,
        "evaluation_metrics": evaluation_metrics,
        "log_history": [dict(item) for item in trainer.state.log_history],
        "length_filter": {
            "action": str(trainer_config["overlength_action"]),
            "train": train_length_filter,
            "validation": validation_length_filter,
        },
        "vram": _summarize_vram(vram_path, run_id),
        "vram_final": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 4),
            "free_gib": round(torch.cuda.mem_get_info()[0] / (1024 ** 3), 4),
            "total_gib": round(torch.cuda.mem_get_info()[1] / (1024 ** 3), 4),
        },
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_fingerprint": checkpoint_fingerprint,
    }
    if split_statistics is not None:
        result["training_split"] = training_split_contract
        result["validation_identity"] = split_statistics["outputs"]["validation"]
        result["base_model"] = base_identity
    if not export:
        return result

    output = config["output"]
    adapter_dir = root / str(output["adapter"])
    merged_dir = root / str(output["merged"])
    gguf_dir = root / str(output["gguf"])
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    # The installed Unsloth exporter writes merged safetensors to the supplied
    # directory and quantizations to <directory>_gguf. Reuse merged_dir and
    # relocate only after both requested GGUF variants have been verified.
    gguf_export = model.save_pretrained_gguf(
        merged_dir,
        processor,
        quantization_method=list(output["gguf_quantizations"]),
    )
    finalized_gguf = finalize_gguf_export(
        gguf_export,
        configured_dir=gguf_dir,
        merged_dir=merged_dir,
        root=root,
        required_quantizations=output["gguf_quantizations"],
    )
    if split_statistics is None:
        raise PipelineError("Full export requires verified split statistics")
    manifest_path = root / str(output["manifest"])
    export_manifest = {
        "version": 1,
        "created_at": utc_now_iso(),
        "base_model": result["base_model"],
        "training_split": result["training_split"],
        "training_length_filter": result["length_filter"],
        "training_contract": {
            "fingerprint": checkpoint_fingerprint,
            "contract": checkpoint_contract,
        },
        "artifacts": {
            "adapter": artifact_files_manifest(adapter_dir, root),
            "merged": artifact_files_manifest(merged_dir, root),
            "gguf": artifact_files_manifest(gguf_dir, root),
        },
        "gguf_quantizations": list(output["gguf_quantizations"]),
    }
    atomic_write_json(manifest_path, export_manifest)
    result["exports"] = {
        "adapter": str(adapter_dir),
        "merged": str(merged_dir),
        "gguf": str(gguf_dir),
        "quantizations": list(finalized_gguf["quantizations"]),
        "manifest": str(manifest_path),
    }
    return result


def run_training(
    *, config_path: Path = Path("config/training.yaml"), smoke_test: bool = False,
    export: bool = True, preflight_only: bool = False, resume_from_latest: bool = False,
) -> dict[str, Any]:
    if resume_from_latest and (smoke_test or preflight_only):
        raise PipelineError("Checkpoint resume requires a full training run")
    if preflight_only:
        return run_training_preflight(config_path=config_path, smoke_test=smoke_test)
    root = Path.cwd()
    config = load_yaml(config_path)
    validate_training_config(config)
    if resume_from_latest and config.get("runtime", {}).get("accelerator") != "rocm":
        raise PipelineError("Explicit checkpoint resume is only enabled for ROCm")
    attempts = [build_attempt(config, root, smoke_test=smoke_test)]
    if not smoke_test and not resume_from_latest and config["trainer"]["oom_fallback"]["enabled"]:
        attempts.extend(
            build_attempt(config, root, smoke_test=False, fallback_index=index)
            for index in range(len(config["trainer"]["oom_fallback"]["profiles"]))
        )
    oom_state_path = root / str(config["output"]["metrics"]) / "oom_state.json"
    oom_contract = _oom_state_contract(config, root)
    known_oom_profiles = [] if smoke_test else _load_oom_state(
        oom_state_path, contract=oom_contract
    )
    all_attempts = attempts
    attempts = [
        attempt
        for index, attempt in enumerate(all_attempts)
        if index == len(all_attempts) - 1 or attempt.profile not in known_oom_profiles
    ]
    skipped_profiles = [
        attempt.profile for attempt in all_attempts
        if attempt not in attempts and attempt.profile in known_oom_profiles
    ]
    if skipped_profiles:
        LOGGER.warning("Skipping profiles with persisted CUDA OOM: %s", skipped_profiles)
    oom_profiles: list[str] = list(skipped_profiles)
    result = None
    for index, attempt in enumerate(attempts):
        try:
            result = _train_once(
                config,
                root,
                attempt,
                export=export and not smoke_test,
                resume_from_latest=resume_from_latest,
            )
            if not smoke_test and (attempt.profile != "primary" or oom_profiles):
                result["oom_fallback_used"] = True
                result["oom_failed_profiles"] = oom_profiles
            if oom_state_path.is_file():
                oom_state_path.unlink()
            break
        except RuntimeError as exc:
            if smoke_test or not _is_cuda_oom(exc) or index + 1 >= len(attempts):
                raise
            oom_profiles.append(attempt.profile)
            _write_oom_state(
                oom_state_path,
                contract=oom_contract,
                failed_profiles=oom_profiles,
            )
            LOGGER.warning(
                "CUDA OOM with profile %s; retrying profile %s",
                attempt.profile,
                attempts[index + 1].profile,
            )
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()
    if result is None:
        raise PipelineError("Training attempts ended without a result")
    report_name = str(config["smoke_test"].get("result_name", "smoke_result.json")) if smoke_test else "training_result.json"
    if Path(report_name).name != report_name:
        raise PipelineError("Training result_name must be a filename")
    report_path = (root / str(config["output"]["metrics"])) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result
