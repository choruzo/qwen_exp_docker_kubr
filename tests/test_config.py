from pathlib import Path

import pytest

from docker_k8s_finetune.config import (
    ConfigError,
    load_all,
    validate_auxiliary_configs,
    validate_cross_config,
    validate_sources_config,
    validate_splits_config,
)


def test_repository_configs_are_valid_and_approved() -> None:
    configs = load_all(Path("config"))
    auxiliary = validate_auxiliary_configs(Path("config"))
    validate_cross_config(configs.sources, auxiliary)
    assert configs.sources["mass_extraction_allowed"] is True
    assert len(configs.sources["sources"]) == 18
    ratios = configs.splits["ratios"]
    assert sum(ratios[name] for name in ("train", "validation", "test")) == pytest.approx(1.0)
    assert configs.training["model"]["name"] == "unsloth/Qwen3.5-4B"
    assert configs.training["trainer"]["overlength_action"] == "exclude"
    assert auxiliary["normalization"]["reverse_instruction"]["max_new_tokens"] == 512
    assert auxiliary["validation"]["hadolint"]["failure_threshold"] == "error"
    assert auxiliary["validation"]["hadolint"]["batch_size"] == 200
    assert auxiliary["benchmark"]["generation"]["max_truncation_rate"] == 0.01
    assert auxiliary["benchmark"]["generation"]["max_new_tokens"] == 2048
    assert auxiliary["benchmark"]["generation"]["batch_size"] == 4
    assert auxiliary["benchmark"]["generation"]["batch_policy"] == "left_padding_first_eos_v1"
    assert auxiliary["benchmark"]["generation"]["batch_order"] == "prompt_length_ascending"
    assert auxiliary["benchmark"]["generation"]["batch_resume_policy"] == (
        "deterministic_full_group_v1"
    )
    assert auxiliary["benchmark"]["generation"]["truncation_detection"] == (
        "max_tokens_without_eos_v1"
    )
    assert auxiliary["benchmark"]["inputs"]["out_of_domain_count"] == 50
    assert auxiliary["benchmark"]["generation"]["cache_compatible_prior_max_new_tokens"] == [1536]
    assert auxiliary["benchmark"]["llm_judge"]["model_identity"]["sha256"] == (
        "eb0f252863d14f7782122a4ac7e8744ed6e4a9fc132584d94686a9441f7c5d35"
    )


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


def test_every_final_source_requires_dedupe_priority() -> None:
    sources = {"sources": {"new_source": {"tier": 2}}}
    auxiliary = {"dedupe": {"approximate": {"source_tiers": {}}}}
    with pytest.raises(ConfigError, match="new_source"):
        validate_cross_config(sources, auxiliary)
