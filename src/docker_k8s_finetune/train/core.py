from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import PipelineError
from ..io import content_hash, file_sha256, stable_json
from ..schema import ChatRecord


@dataclass(frozen=True)
class TrainingAttempt:
    max_seq_length: int
    batch_size: int
    gradient_accumulation_steps: int
    output_dir: Path
    max_steps: int = -1
    smoke_test: bool = False
    profile: str = "primary"

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
    if config.get("runtime", {}).get("require_local_model"):
        raise PipelineError(f"ROCm profile requires the complete local base model: {local}")
    return str(model["name"])


def verify_local_model_provenance(model_path: Path, model_config: Mapping[str, Any]) -> dict[str, Any]:
    expected = model_config.get("expected_sha256") or {}
    if not isinstance(expected, Mapping) or not expected:
        raise PipelineError("Local model requires expected_sha256 provenance")
    index_path = model_path / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read local model index for provenance: {exc}") from exc
    shards = set(index.get("weight_map", {}).values())
    missing_hashes = shards - set(map(str, expected))
    if missing_hashes:
        raise PipelineError(f"Local model shards lack expected hashes: {sorted(missing_hashes)}")

    actual: dict[str, str] = {}
    root = model_path.resolve()
    for name, expected_hash in expected.items():
        path = (model_path / str(name)).resolve()
        if path.parent != root or not path.is_file():
            raise PipelineError(f"Invalid or missing model provenance file: {name}")
        actual_hash = file_sha256(path)
        if actual_hash != str(expected_hash).lower():
            raise PipelineError(f"Local model hash mismatch: {name}")
        actual[str(name)] = actual_hash

    expected_revision = str(model_config.get("revision", "")).strip()
    observed_revision = None
    if (model_path / ".git").exists() and expected_revision:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if completed.returncode != 0:
            raise PipelineError(f"Cannot resolve local model Git revision: {completed.stderr.strip()}")
        observed_revision = completed.stdout.strip()
        if observed_revision != expected_revision:
            raise PipelineError(
                f"Local model revision mismatch: {observed_revision} != {expected_revision}"
            )
    return {"revision": observed_revision or expected_revision, "sha256": actual}


def artifact_files_manifest(directory: Path, root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    resolved_directory = directory.resolve()
    if not resolved_directory.is_relative_to(resolved_root):
        raise PipelineError(f"Artifact directory is outside project root: {resolved_directory}")
    files = []
    for path in sorted(resolved_directory.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_directory):
            raise PipelineError(f"Artifact file escapes its directory: {resolved}")
        files.append({
            "path": resolved.relative_to(resolved_root).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": file_sha256(resolved),
        })
    if not files:
        raise PipelineError(f"Export artifact directory is empty: {directory}")
    return files


def finalize_gguf_export(
    export_result: Mapping[str, Any],
    *,
    configured_dir: Path,
    merged_dir: Path,
    root: Path,
    required_quantizations: Iterable[str],
) -> dict[str, Any]:
    """Move Unsloth's ``<merged>_gguf`` output to the configured directory."""
    resolved_root = root.resolve()
    expected_source = Path(f"{merged_dir}_gguf").resolve()
    reported_source = Path(str(export_result.get("gguf_directory", ""))).resolve()
    if reported_source != expected_source:
        raise PipelineError(
            f"Unexpected Unsloth GGUF directory: {reported_source} != {expected_source}"
        )
    if not reported_source.is_relative_to(resolved_root) or not reported_source.is_dir():
        raise PipelineError(f"Unsloth GGUF directory is unsafe or missing: {reported_source}")

    reported_files = export_result.get("gguf_files")
    if not isinstance(reported_files, list) or not reported_files:
        raise PipelineError("Unsloth did not report any GGUF files")
    files: list[Path] = []
    for value in reported_files:
        path = Path(str(value)).resolve()
        if not path.is_relative_to(reported_source) or not path.is_file() or path.suffix.lower() != ".gguf":
            raise PipelineError(f"Invalid GGUF file reported by Unsloth: {value}")
        files.append(path)

    normalized_names = [
        path.name.replace("_", "").replace("-", "").casefold() for path in files
    ]
    required = [str(value) for value in required_quantizations]
    missing = [
        method
        for method in required
        if method.replace("_", "").replace("-", "").casefold()
        not in " ".join(normalized_names)
    ]
    if missing:
        raise PipelineError(f"Unsloth did not create required GGUF quantizations: {missing}")

    destination = configured_dir.resolve()
    if not destination.is_relative_to(resolved_root):
        raise PipelineError(f"Configured GGUF directory is outside project root: {destination}")
    if destination != reported_source:
        if destination.exists():
            raise PipelineError(f"Configured GGUF directory already exists: {destination}")
        reported_source.rename(destination)
    relocated = [destination / path.name for path in files]
    if not all(path.is_file() for path in relocated):
        raise PipelineError("GGUF relocation did not preserve every exported file")
    return {
        "directory": destination,
        "files": relocated,
        "quantizations": required,
    }


def verify_export_artifact(root: Path, manifest_path: Path, artifact: str) -> dict[str, Any]:
    path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    if not path.is_file():
        raise PipelineError(f"Export manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read export manifest: {exc}") from exc
    entries = manifest.get("artifacts", {}).get(artifact)
    if not isinstance(entries, list) or not entries:
        raise PipelineError(f"Export manifest has no artifact section: {artifact}")
    resolved_root = root.resolve()
    for entry in entries:
        file_path = (root / str(entry["path"])).resolve()
        if not file_path.is_relative_to(resolved_root) or not file_path.is_file():
            raise PipelineError(f"Export artifact file is unsafe or missing: {entry.get('path')}")
        if file_path.stat().st_size != int(entry["bytes"]):
            raise PipelineError(f"Export artifact size mismatch: {entry['path']}")
        if file_sha256(file_path) != str(entry["sha256"]):
            raise PipelineError(f"Export artifact hash mismatch: {entry['path']}")
    return {
        "manifest": path.relative_to(resolved_root).as_posix(),
        "artifact": artifact,
        "file_count": len(entries),
        "files": [dict(entry) for entry in entries],
        "base_model": manifest.get("base_model"),
        "split_test_sha256": manifest.get("training_split", {}).get("test"),
    }


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


def resolve_text_tokenizer(processor: Any) -> Any:
    """Return the text tokenizer without routing strings through a vision processor.

    ``FastVisionModel`` returns a multimodal processor for Qwen3.5. Its first
    positional argument is ``images``, while TRL tokenizes the configured text
    field positionally. Passing that processor directly therefore makes valid
    ChatML look like an image path/base64 payload. Text-only fine-tuning must
    explicitly use the processor's nested tokenizer.
    """
    tokenizer = getattr(processor, "tokenizer", None) or processor
    if not callable(tokenizer) or not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise PipelineError("Qwen3.5 processor does not expose a usable text tokenizer")
    return tokenizer


def filter_formatted_by_token_length(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_length <= 0 or batch_size <= 0:
        raise PipelineError("Token-length audit limits must be positive")
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    lengths: list[int] = []
    kept_hashes: list[str] = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        encoded = tokenizer(
            [str(record["text"]) for record in batch],
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list) or len(input_ids) != len(batch):
            raise PipelineError("Tokenizer returned invalid batched input_ids for length audit")
        for record, tokens in zip(batch, input_ids):
            length = len(tokens)
            lengths.append(length)
            meta = record.get("meta", {})
            identifier = str(meta.get("content_hash") or content_hash(str(record["text"])))
            if length > max_length:
                excluded.append({
                    "content_hash": identifier,
                    "category": str(meta.get("category", "unknown")),
                    "source": str(meta.get("source", "unknown")),
                    "tokens": length,
                })
            else:
                kept.append(record)
                kept_hashes.append(identifier)
    if not kept:
        raise PipelineError("Token-length filter removed the complete dataset")
    ordered = sorted(lengths)
    p99_index = round((len(ordered) - 1) * 0.99) if ordered else 0
    report = {
        "max_length": max_length,
        "input_records": len(records),
        "kept_records": len(kept),
        "excluded_records": len(excluded),
        "excluded_rate": len(excluded) / len(records) if records else 0.0,
        "maximum_observed_tokens": max(lengths) if lengths else None,
        "p99_tokens": ordered[p99_index] if ordered else None,
        "kept_content_hashes_sha256": content_hash(stable_json(kept_hashes)),
        "excluded": excluded,
    }
    return kept, report


def latest_checkpoint(output_dir: Path, *, expected_fingerprint: str | None = None) -> str | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for path in output_dir.glob("checkpoint-*"):
            if path.is_dir():
                try:
                    step = int(path.name.rsplit("-", 1)[1])
                except ValueError:
                    continue
                required = (
                    path / "trainer_state.json",
                    path / "optimizer.pt",
                    path / "scheduler.pt",
                    path / "rng_state.pth",
                )
                weights = tuple(
                    path / name
                    for name in (
                        "adapter_model.safetensors",
                        "model.safetensors",
                        "pytorch_model.bin",
                    )
                )
                if any(not item.is_file() or item.stat().st_size <= 0 for item in required):
                    continue
                if not any(item.is_file() and item.stat().st_size > 0 for item in weights):
                    continue
                try:
                    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if int(state.get("global_step", -1)) != step:
                    continue
                if expected_fingerprint is not None:
                    provenance_path = path / "checkpoint_provenance.json"
                    try:
                        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if provenance.get("fingerprint") != expected_fingerprint:
                        continue
                candidates.append((step, path))
    return str(max(candidates)[1]) if candidates else None


def build_attempt(
    config: Mapping[str, Any], root: Path, *, smoke_test: bool,
    fallback_index: int | None = None,
) -> TrainingAttempt:
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
            profile="smoke",
        )
    if fallback_index is not None:
        profiles = trainer["oom_fallback"]["profiles"]
        try:
            fallback_config = profiles[fallback_index]
        except (IndexError, TypeError) as exc:
            raise PipelineError(f"Unknown OOM fallback profile index: {fallback_index}") from exc
        return TrainingAttempt(
            max_seq_length=int(fallback_config["max_seq_length"]),
            batch_size=int(fallback_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(fallback_config["gradient_accumulation_steps"]),
            output_dir=(
                root / str(config["output"]["checkpoints"]) / str(fallback_config["name"])
            ),
            profile=str(fallback_config["name"]),
        )
    return TrainingAttempt(
        max_seq_length=int(config["model"]["max_seq_length"]),
        batch_size=int(trainer["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(trainer["gradient_accumulation_steps"]),
        output_dir=root / str(config["output"]["checkpoints"]) / "primary",
        profile="primary",
    )


def validate_training_config(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    architecture = config.get("architecture", {})
    lora = config.get("lora", {})
    trainer = config.get("trainer", {})
    runtime = config.get("runtime", {})
    accelerator = runtime.get("accelerator", "cuda")
    if accelerator not in ("cuda", "rocm"):
        raise PipelineError("Training accelerator must be cuda or rocm")
    if accelerator == "rocm":
        if model.get("load_in_4bit") is not False or model.get("dtype") != "bfloat16":
            raise PipelineError("ROCm profile requires unquantized BF16 LoRA")
        if trainer.get("optimizer") != "adamw_torch":
            raise PipelineError("ROCm profile requires the native torch AdamW optimizer")
        if trainer.get("resume_from_checkpoint") != "none":
            raise PipelineError("ROCm profile must start without automatic checkpoint resume")
        output = config.get("output", {})
        for key in ("checkpoints", "adapter", "merged", "gguf", "metrics", "manifest"):
            if not str(output.get(key, "")).startswith("artifacts/rocm/"):
                raise PipelineError(f"ROCm output {key} must be isolated under artifacts/rocm")
    if model.get("loader") != "FastVisionModel" or model.get("architecture") != "qwen3_5_vlm_text_only":
        raise PipelineError("Qwen3.5-4B must use FastVisionModel in text-only mode")
    revision = str(model.get("revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision.lower()):
        raise PipelineError("Base model revision must be a full 40-character commit")
    expected_hashes = model.get("expected_sha256") or {}
    if not expected_hashes or any(
        len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value).lower())
        for value in expected_hashes.values()
    ):
        raise PipelineError("Base model expected_sha256 entries must be complete SHA-256 hashes")
    if architecture.get("text_only") is not True or architecture.get("finetune_vision_layers") is not False:
        raise PipelineError("Text-only training must keep vision layers frozen")
    if lora.get("target_modules") != "auto":
        raise PipelineError("Qwen3.5 hybrid attention requires Unsloth automatic module selection")
    if trainer.get("response_only_loss") is not True:
        raise PipelineError("Response-only loss is required")
    if trainer.get("train_sampling_strategy") != "random":
        raise PipelineError(
            "Batch-one training must use random sampling to avoid concentrating long sequences"
        )
    if trainer.get("overlength_action") != "exclude":
        raise PipelineError("Overlength training examples must be excluded explicitly")
    if int(trainer.get("length_audit_batch_size", 0)) <= 0:
        raise PipelineError("length_audit_batch_size must be positive")
    if int(trainer.get("per_device_eval_batch_size", 0)) <= 0:
        raise PipelineError("per_device_eval_batch_size must be positive")
    if int(trainer.get("eval_steps", 0)) <= 0:
        raise PipelineError("eval_steps must be positive")
    fallback = trainer.get("oom_fallback", {})
    profiles = fallback.get("profiles")
    if fallback.get("enabled") is not True or not isinstance(profiles, list) or len(profiles) < 2:
        raise PipelineError("At least two ordered OOM fallback profiles are required")
    names = [str(profile.get("name", "")) for profile in profiles if isinstance(profile, Mapping)]
    if len(names) != len(profiles) or any(not name for name in names) or len(set(names)) != len(names):
        raise PipelineError("OOM fallback profile names must be non-empty and unique")
    previous_length = int(model["max_seq_length"])
    previous_batch_size = int(trainer["per_device_train_batch_size"])
    for profile in profiles:
        sequence_length = int(profile.get("max_seq_length", 0))
        batch_size = int(profile.get("per_device_train_batch_size", 0))
        accumulation = int(profile.get("gradient_accumulation_steps", 0))
        if (
            sequence_length <= 0
            or sequence_length > previous_length
            or batch_size <= 0
            or batch_size > previous_batch_size
            or accumulation <= 0
            or (sequence_length == previous_length and batch_size == previous_batch_size)
        ):
            raise PipelineError("OOM fallback profiles must reduce memory monotonically")
        if batch_size * accumulation != int(trainer["effective_batch_size"]):
            raise PipelineError("OOM fallback profiles must preserve effective batch size")
        previous_length = sequence_length
        previous_batch_size = batch_size
    if int(trainer.get("effective_batch_size", -1)) != int(trainer["per_device_train_batch_size"]) * int(trainer["gradient_accumulation_steps"]):
        raise PipelineError("Configured effective_batch_size is inconsistent")
