#!/usr/bin/env python3
"""Audit paired frozen ROCm predictions without publishing sample-level data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summary(records: list[dict]) -> dict:
    lengths = sorted(int(record["completion_tokens"]) for record in records if record.get("completion_tokens") is not None)
    similarities = [float(record["semantic_similarity"]) for record in records if record.get("semantic_similarity") is not None]
    return {
        "count": len(records),
        "retry_count": sum(int(record.get("retry_count") or 0) > 0 for record in records),
        "truncated_count": sum(record.get("truncated") is True for record in records),
        "exact_match_count": sum(float(record.get("exact_match") or 0) == 1 for record in records),
        "mean_similarity": mean(similarities) if similarities else None,
        "median_completion_tokens": median(lengths) if lengths else None,
        "p95_completion_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))] if lengths else None,
    }


def diagnose(baseline: dict, finetuned: dict) -> dict:
    if baseline.get("split_test_sha256") != finetuned.get("split_test_sha256"):
        raise ValueError("Frozen test SHA differs between variants")
    if baseline.get("generation") != finetuned.get("generation"):
        raise ValueError("Generation settings differ between variants")
    base_records = baseline["records"]
    fine_records = finetuned["records"]
    if len(base_records) != len(fine_records):
        raise ValueError("Record counts differ")
    keys = ("content_hash", "category", "question", "reference")
    for position, (base, fine) in enumerate(zip(base_records, fine_records)):
        if any(base.get(key) != fine.get(key) for key in keys):
            raise ValueError(f"Paired record identity differs at position {position}")
    if len({record["content_hash"] for record in base_records}) != len(base_records):
        raise ValueError("Duplicate record content hash")
    by_category = defaultdict(lambda: ([], []))
    for base, fine in zip(base_records, fine_records):
        pair = by_category[base["category"]]
        pair[0].append(base)
        pair[1].append(fine)
    transitions = Counter(
        (bool(base.get("truncated")), bool(fine.get("truncated")))
        for base, fine in zip(base_records, fine_records)
    )
    retry_transitions = Counter(
        (bool(base.get("retry_count")), bool(fine.get("retry_count")))
        for base, fine in zip(base_records, fine_records)
    )
    return {
        "version": 1,
        "test_sha256": baseline["split_test_sha256"],
        "generation_gate": {
            "max_truncation_rate": baseline["generation"]["max_truncation_rate"],
            "baseline_passed": baseline["truncation_audit"]["passed"],
            "finetuned_passed": finetuned["truncation_audit"]["passed"],
        },
        "overall": {"baseline": _summary(base_records), "finetuned": _summary(fine_records)},
        "by_category": {
            category: {"baseline": _summary(pair[0]), "finetuned": _summary(pair[1])}
            for category, pair in sorted(by_category.items())
        },
        "paired_transitions": {
            "new_truncations": transitions[(False, True)],
            "resolved_truncations": transitions[(True, False)],
            "both_truncated": transitions[(True, True)],
            "new_retries": retry_transitions[(False, True)],
            "resolved_retries": retry_transitions[(True, False)],
            "both_retried": retry_transitions[(True, True)],
        },
        "retry_failure_rate": {
            "baseline": _rate(sum(r.get("truncated") is True for r in base_records), sum(bool(r.get("retry_count")) for r in base_records)),
            "finetuned": _rate(sum(r.get("truncated") is True for r in fine_records), sum(bool(r.get("retry_count")) for r in fine_records)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("finetuned", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(_load(args.baseline), _load(args.finetuned)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
