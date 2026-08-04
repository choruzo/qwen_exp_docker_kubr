from pathlib import Path

import pytest

from docker_k8s_finetune.config import ConfigError, load_all, validate_sources_config, validate_splits_config


def test_repository_configs_are_valid_and_approved() -> None:
    configs = load_all(Path("config"))
    assert configs.sources["mass_extraction_allowed"] is True
    assert len(configs.sources["sources"]) == 18
    ratios = configs.splits["ratios"]
    assert sum(ratios[name] for name in ("train", "validation", "test")) == pytest.approx(1.0)
    assert configs.training["model"]["name"] == "unsloth/Qwen3.5-4B"


def test_sources_gate_rejects_unapproved_extraction() -> None:
    config = {
        "version": 1,
        "status": "proposed",
        "mass_extraction_allowed": False,
        "intermediate_record": {"allowed_categories": []},
        "sources": {},
    }
    with pytest.raises(ConfigError, match="approved"):
        validate_sources_config(config)


def test_split_ratios_must_sum_to_one() -> None:
    config = {
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.2},
        "category_constraints": {"required_categories": []},
        "post_split_leakage_audit": {"threshold": 0.92},
    }
    with pytest.raises(ConfigError, match="sum to 1.0"):
        validate_splits_config(config)
