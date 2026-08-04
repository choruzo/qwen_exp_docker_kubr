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
    benchmark = loaded["benchmark"]
    required_variants = {"baseline", "finetuned_safetensors", "finetuned_gguf"}
    if set(benchmark.get("variants", {})) != required_variants:
        raise ConfigError("Benchmark variants are incomplete")
    return loaded


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
