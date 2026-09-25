#!/usr/bin/env python3
"""Score two paired validation probes without exposing sample-level text."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from docker_k8s_finetune.benchmark.scoring import semantic_scores
from docker_k8s_finetune.config import load_yaml
from docker_k8s_finetune.io import file_sha256


def _summary(records: list[dict], scores: list[float]) -> dict:
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_category[str(record["category"])].append(index)

    def aggregate(indexes: list[int]) -> dict:
        return {
            "count": len(indexes),
            "semantic_similarity_mean": statistics.mean(scores[index] for index in indexes),
            "completion_tokens_median": statistics.median(
                int(records[index]["completion_tokens"]) for index in indexes
            ),
            "retries": sum(int(records[index]["retry_count"]) for index in indexes),
            "truncated": sum(bool(records[index]["truncated"]) for index in indexes),
        }

    return {
        "overall": aggregate(list(range(len(records)))),
        "by_category": {
            category: aggregate(indexes)
            for category, indexes in sorted(by_category.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("alternative", type=Path)
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("config/benchmark.rocm.best_epoch_v2.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark_path = root / args.benchmark_config
    benchmark = load_yaml(benchmark_path)
    probes = [
        json.loads((root / path).read_text(encoding="utf-8"))
        for path in (args.frozen, args.alternative)
    ]
    if probes[0].get("frozen_benchmark_sha256") != file_sha256(benchmark_path):
        raise ValueError("Probe does not match the benchmark config")
    if any(probe.get("selected_hashes") != probes[0].get("selected_hashes") for probe in probes[1:]):
        raise ValueError("Probe selections differ")
    records = [probe["records"] for probe in probes]
    hashes = [[record["content_hash"] for record in group] for group in records]
    if len(records[0]) != len(records[1]) or hashes[0] != hashes[1]:
        raise ValueError("Probe records are not complete paired observations")
    references = [record["reference"] for group in records for record in group]
    predictions = [record["prediction"] for group in records for record in group]
    scores = semantic_scores(references, predictions, benchmark["semantic_similarity"])
    size = len(records[0])
    frozen_scores, alternative_scores = scores[:size], scores[size:]
    deltas = [alternative - frozen for frozen, alternative in zip(frozen_scores, alternative_scores)]
    payload = {
        "version": 1,
        "benchmark_config_sha256": file_sha256(benchmark_path),
        "frozen_result_sha256": file_sha256(root / args.frozen),
        "alternative_result_sha256": file_sha256(root / args.alternative),
        "frozen": _summary(records[0], frozen_scores),
        "alternative": _summary(records[1], alternative_scores),
        "paired": {
            "semantic_similarity_mean_delta": statistics.mean(deltas),
            "semantic_similarity_median_delta": statistics.median(deltas),
            "alternative_wins": sum(delta > 0 for delta in deltas),
            "frozen_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
