from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


def deterministic_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def allocate_stratum(
    groups: list[tuple[str, int]], *, seed: int, validation_ratio: float, test_ratio: float,
    minimum_eval: int, rare_threshold: int,
) -> dict[str, str]:
    ordered = sorted(groups, key=lambda item: deterministic_key(seed, item[0]))
    total = sum(size for _, size in ordered)
    if total < 3:
        return {group: "train" for group, _ in ordered}
    minimum = 1 if total < rare_threshold else minimum_eval
    target_test = max(minimum, round(total * test_ratio))
    target_validation = max(minimum, round(total * validation_ratio))
    if target_test + target_validation >= total:
        target_test = 1
        target_validation = 1
    assignments: dict[str, str] = {}
    counts = {"test": 0, "validation": 0}
    for group, size in ordered:
        if counts["test"] < target_test:
            split = "test"
        elif counts["validation"] < target_validation:
            split = "validation"
        else:
            split = "train"
        assignments[group] = split
        if split in counts:
            counts[split] += size
    return assignments


def assistant_text(record: Mapping[str, Any]) -> str:
    return next(
        (str(message.get("content", "")) for message in record.get("messages", []) if message.get("role") == "assistant"),
        "",
    )


def hard_complexity(record: Mapping[str, Any]) -> tuple[int, list[str]]:
    text = " ".join(
        str(message.get("content", "")) for message in record.get("messages", []) if isinstance(message, Mapping)
    ).casefold()
    features: list[str] = []
    component_hits = sum(word in text for word in ("kubelet", "coredns", "etcd", "ingress", "containerd", "scheduler"))
    if component_hits >= 2:
        features.append("multiple_components")
    if re.search(r"\b(first|then|next|step|after|before|investigate|diagnos)\b", text):
        features.append("multi_step_diagnosis")
    if re.search(r"\b(error|failed|event|log|traceback|exception)\b", text) or "```" in text:
        features.append("contains_logs_or_events")
    if re.search(r"\b(oom|memory|cpu|limit|request|evict|throttl)\b", text):
        features.append("requires_resource_constraints_reasoning")
    if re.search(r"\b(dns|network|cni|service|ingress|endpoint|connection)\b", text):
        features.append("networking_or_dns_edge_case")
    if re.search(r"\b(storage|volume|pvc|etcd|mount|filesystem)\b", text):
        features.append("storage_or_etcd_edge_case")
    if re.search(r"\b(but|although|despite|intermittent|only one|works locally)\b", text):
        features.append("misleading_symptom")
    return len(features), features


def split_statistics(records_by_split: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name, records_iterable in records_by_split.items():
        records = list(records_iterable)
        categories: dict[str, int] = defaultdict(int)
        sources: dict[str, int] = defaultdict(int)
        for record in records:
            meta = record.get("meta", {})
            categories[str(meta.get("category"))] += 1
            sources[str(meta.get("source"))] += 1
        result[split_name] = {
            "count": len(records),
            "categories": dict(sorted(categories.items())),
            "sources": dict(sorted(sources.items())),
        }
    return result
