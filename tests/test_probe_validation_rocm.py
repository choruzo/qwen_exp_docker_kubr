"""Tests for deterministic validation-probe selection."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from docker_k8s_finetune.config import REQUIRED_CATEGORIES


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_validation_rocm.py"
SPEC = importlib.util.spec_from_file_location("probe_validation_rocm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _records() -> list[dict]:
    return [
        {
            "meta": {"category": category, "content_hash": f"{category}-{length}"},
            "messages": [{"role": "assistant", "content": "x" * length}],
        }
        for category in REQUIRED_CATEGORIES
        for length in (1, 2, 3, 4)
    ]


class ValidationProbeSelectionTests(unittest.TestCase):
    def test_selects_median_and_p90_per_category(self) -> None:
        selected = probe.select_validation_records(_records())
        self.assertEqual(len(selected), 12)
        for category in REQUIRED_CATEGORIES:
            lengths = sorted(
                len(record["messages"][0]["content"])
                for record in selected
                if record["meta"]["category"] == category
            )
            self.assertEqual(lengths, [2, 4])

    def test_rejects_duplicate_hash(self) -> None:
        records = _records()
        records[1]["meta"]["content_hash"] = records[0]["meta"]["content_hash"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.select_validation_records(records)

    def test_selects_even_quantiles_per_category(self) -> None:
        selected = probe.select_validation_records_quantiles(_records(), per_category=3)
        self.assertEqual(len(selected), 18)
        for category in REQUIRED_CATEGORIES:
            lengths = sorted(
                len(record["messages"][0]["content"])
                for record in selected
                if record["meta"]["category"] == category
            )
            self.assertEqual(lengths, [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
