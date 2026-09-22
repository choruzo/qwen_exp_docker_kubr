"""Checks for the privacy-preserving paired ROCm diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_rocm_benchmark.py"
SPEC = importlib.util.spec_from_file_location("diagnose_rocm_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def _payload(*, truncated: bool, retry: int = 0) -> dict:
    return {
        "split_test_sha256": "same-sha",
        "generation": {"max_truncation_rate": 0.01},
        "truncation_audit": {"passed": not truncated},
        "records": [{
            "content_hash": "one",
            "category": "concepto",
            "question": "question",
            "reference": "answer",
            "completion_tokens": 2048 if truncated else 4,
            "semantic_similarity": 0.5,
            "exact_match": 0.0,
            "retry_count": retry,
            "truncated": truncated,
            "prediction": "private response",
        }],
    }


def test_diagnose_counts_paired_transition_without_exposing_responses() -> None:
    result = diagnostic.diagnose(_payload(truncated=False), _payload(truncated=True, retry=1))
    assert result["paired_transitions"]["new_truncations"] == 1
    assert result["paired_transitions"]["new_retries"] == 1
    assert result["retry_failure_rate"]["finetuned"] == 1.0
    assert "private response" not in str(result)


@pytest.mark.parametrize("field,value", [("content_hash", "different"), ("reference", "different")])
def test_diagnose_rejects_unpaired_records(field: str, value: str) -> None:
    finetuned = _payload(truncated=False)
    finetuned["records"][0][field] = value
    with pytest.raises(ValueError, match="Paired record identity"):
        diagnostic.diagnose(_payload(truncated=False), finetuned)


def test_diagnose_rejects_different_generation_settings() -> None:
    finetuned = _payload(truncated=False)
    finetuned["generation"]["max_new_tokens"] = 4096
    with pytest.raises(ValueError, match="Generation settings"):
        diagnostic.diagnose(_payload(truncated=False), finetuned)
