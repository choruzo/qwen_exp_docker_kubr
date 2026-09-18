from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError, PipelineError


REQUIRED_CATEGORIES = {
    "concepto",
    "comando_cli",
    "generacion_yaml",
    "troubleshooting",
    "dockerfile",
    "arquitectura",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise ConfigError("PyYAML is required; install the project before running the CLI") from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return loaded


def dotted_get(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Unknown configuration reference: {path}")
        current = current[part]
    return current


def validate_sources_config(config: Mapping[str, Any], *, require_approved: bool = True) -> None:
    if config.get("version") != 1:
        raise ConfigError("config/sources.yaml version must be 1")
    if require_approved and config.get("status") != "approved":
        raise ConfigError("Sources configuration has not been approved")
    if require_approved and config.get("mass_extraction_allowed") is not True:
        raise ConfigError("Mass extraction is disabled in config/sources.yaml")
    categories = set(config.get("intermediate_record", {}).get("allowed_categories", []))
    if categories != REQUIRED_CATEGORIES:
        raise ConfigError(f"Allowed categories must be exactly {sorted(REQUIRED_CATEGORIES)}")
    sources = config.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ConfigError("At least one source must be configured")
    required = {"kind", "tier"}
    for source_name, source in sources.items():
        if not isinstance(source, dict):
            raise ConfigError(f"Source {source_name!r} must be a mapping")
        missing = required - source.keys()
        if missing:
            raise ConfigError(f"Source {source_name!r} is missing keys: {sorted(missing)}")
        if "license" not in source and "license_strategy" not in source:
            raise ConfigError(f"Source {source_name!r} must define license or license_strategy")


def validate_splits_config(config: Mapping[str, Any], *, require_approved: bool = True) -> None:
    ratios = config.get("ratios", {})
    values = [ratios.get(name) for name in ("train", "validation", "test")]
    if not all(isinstance(value, (int, float)) for value in values):
        raise ConfigError("Split ratios must be numeric")
    if abs(sum(values) - 1.0) > 1e-12:
        raise ConfigError("Split ratios must sum to 1.0")
    required = set(config.get("category_constraints", {}).get("required_categories", []))
    if required != REQUIRED_CATEGORIES:
        raise ConfigError("Split category constraints do not match the normalized schema")
    threshold = config.get("post_split_leakage_audit", {}).get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
        raise ConfigError("Leakage similarity threshold must be in (0, 1]")
    if require_approved and config.get("status") != "approved":
        raise ConfigError("Split configuration has not been approved")


def validate_training_config(config: Mapping[str, Any]) -> None:
    from .train.core import validate_training_config as validate

    try:
        validate(config)
    except PipelineError as exc:
        raise ConfigError(str(exc)) from exc


def validate_auxiliary_configs(config_dir: str | Path = "config") -> dict[str, Any]:
    root = Path(config_dir)
    names = ("cleaning", "normalization", "reverse_instruction_review", "dedupe", "validation", "benchmark")
    loaded = {name: load_yaml(root / f"{name}.yaml") for name in names}
    for name, config in loaded.items():
        if config.get("version") != 1:
            raise ConfigError(f"config/{name}.yaml version must be 1")
    reverse = loaded["normalization"].get("reverse_instruction", {})
    if int(reverse.get("max_new_tokens", 0)) <= 0:
        raise ConfigError("Reverse-instruction max_new_tokens must be positive")
    dedupe = loaded["dedupe"]["approximate"]
    revision = str(dedupe.get("model_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision.lower()):
        raise ConfigError("Embedding model_revision must be a full 40-character commit")
    if not dedupe.get("embeddings"):
        raise ConfigError("Deduplication must persist its semantic embedding cache")
    validation = loaded["validation"]
    for tool in ("kubeconform", "hadolint"):
        image = str(validation[tool].get("image", ""))
        if "@sha256:" not in image:
            raise ConfigError(f"{tool} image must be pinned by digest")
    threshold = str(validation["hadolint"].get("failure_threshold", ""))
    if threshold not in {"ignore", "none", "style", "info", "warning", "error"}:
        raise ConfigError("hadolint failure_threshold is invalid")
    if int(validation["hadolint"].get("batch_size", 0)) <= 0:
        raise ConfigError("hadolint batch_size must be positive")
    benchmark = loaded["benchmark"]
    required_variants = {"baseline", "finetuned_safetensors", "finetuned_gguf"}
    if set(benchmark.get("variants", {})) != required_variants:
        raise ConfigError("Benchmark variants are incomplete")
    if not isinstance(benchmark.get("inputs", {}).get("out_of_domain_count"), int) or int(
        benchmark["inputs"]["out_of_domain_count"]
    ) <= 0:
        raise ConfigError("Benchmark out_of_domain_count must be a positive integer")
    benchmark_seed = benchmark.get("seed")
    if benchmark.get("generation", {}).get("seed") != benchmark_seed:
        raise ConfigError("Benchmark generation seed must match the top-level seed")
    max_truncation_rate = benchmark.get("generation", {}).get("max_truncation_rate")
    if (
        not isinstance(max_truncation_rate, (int, float))
        or not 0.0 <= float(max_truncation_rate) <= 1.0
    ):
        raise ConfigError("Benchmark max_truncation_rate must be in [0, 1]")
    max_new_tokens = benchmark.get("generation", {}).get("max_new_tokens")
    batch_size = benchmark.get("generation", {}).get("batch_size", 1)
    batch_policy = benchmark.get("generation", {}).get("batch_policy")
    batch_order = benchmark.get("generation", {}).get("batch_order", "manifest")
    batch_resume_policy = benchmark.get("generation", {}).get("batch_resume_policy")
    truncation_detection = benchmark.get("generation", {}).get("truncation_detection")
    compatible_prior_limits = benchmark.get("generation", {}).get(
        "cache_compatible_prior_max_new_tokens", []
    )
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ConfigError("Benchmark max_new_tokens must be a positive integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigError("Benchmark generation batch_size must be a positive integer")
    if batch_size > 1 and batch_policy != "left_padding_first_eos_v1":
        raise ConfigError(
            "Batched benchmark generation requires batch_policy=left_padding_first_eos_v1"
        )
    if batch_size > 1 and batch_order not in {"manifest", "prompt_length_ascending"}:
        raise ConfigError(
            "Batched benchmark generation batch_order must be manifest or prompt_length_ascending"
        )
    if batch_size > 1 and batch_resume_policy != "deterministic_full_group_v1":
        raise ConfigError(
            "Batched benchmark generation requires batch_resume_policy=deterministic_full_group_v1"
        )
    if truncation_detection != "max_tokens_without_eos_v1":
        raise ConfigError(
            "Benchmark truncation_detection must be max_tokens_without_eos_v1"
        )
    if (
        not isinstance(compatible_prior_limits, list)
        or any(
            not isinstance(value, int) or value <= 0 or value >= max_new_tokens
            for value in compatible_prior_limits
        )
        or len(set(compatible_prior_limits)) != len(compatible_prior_limits)
    ):
        raise ConfigError(
            "Benchmark compatible prior token limits must be unique positive integers below max_new_tokens"
        )
    retry = benchmark.get("generation", {}).get("retry_on_truncation", {})
    if (
        retry.get("enabled") is not True
        or retry.get("max_attempts") != 1
        or not isinstance(retry.get("repetition_penalty"), (int, float))
        or float(retry["repetition_penalty"])
        <= float(benchmark["generation"].get("repetition_penalty", 1.0))
    ):
        raise ConfigError(
            "Benchmark truncation retry must use one attempt with a higher repetition penalty"
        )
    if benchmark.get("llm_judge", {}).get("seed") != benchmark_seed:
        raise ConfigError("LLM judge seed must match the benchmark seed")
    judge = benchmark.get("llm_judge", {})
    identity = judge.get("model_identity", {})
    for field in ("alias", "filename", "quantization"):
        if not str(identity.get(field, "")).strip():
            raise ConfigError(f"LLM judge model_identity.{field} is required")
    judge_hash = str(identity.get("sha256", ""))
    if len(judge_hash) != 64 or any(
        character not in "0123456789abcdef" for character in judge_hash.lower()
    ):
        raise ConfigError("LLM judge model_identity.sha256 must be a complete SHA-256")
    if not isinstance(identity.get("bytes"), int) or int(identity["bytes"]) <= 0:
        raise ConfigError("LLM judge model_identity.bytes must be a positive integer")
    if judge.get("require_llama_props") is not True:
        raise ConfigError("LLM judge must require llama-server /props verification")
    return loaded


def validate_cross_config(
    sources: Mapping[str, Any], auxiliary: Mapping[str, Mapping[str, Any]],
) -> None:
    final_sources = {
        str(name)
        for name, source in sources.get("sources", {}).items()
        if source.get("tier") != "quarantine" and source.get("include_in_final", True)
    }
    tiered_sources = set(auxiliary["dedupe"]["approximate"].get("source_tiers", {}))
    missing = final_sources - tiered_sources
    if missing:
        raise ConfigError(
            f"Final sources missing approximate-dedupe priority: {sorted(missing)}"
        )


@dataclass(frozen=True)
class LoadedConfigs:
    sources: dict[str, Any]
    splits: dict[str, Any]
    training: dict[str, Any]


def load_all(config_dir: str | Path = "config") -> LoadedConfigs:
    root = Path(config_dir)
    sources = load_yaml(root / "sources.yaml")
    splits = load_yaml(root / "splits.yaml")
    training = load_yaml(root / "training.yaml")
    validate_sources_config(sources)
    validate_splits_config(splits)
    validate_training_config(training)
    return LoadedConfigs(sources=sources, splits=splits, training=training)
