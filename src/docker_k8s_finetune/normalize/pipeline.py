from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, read_jsonl
from .core import direct_pair_to_chatml


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"version": 1}


def run_direct_normalization(
    *, config_path: Path = Path("config/normalization.yaml"), root: Path = Path(".")
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    input_path = root / config["input"]
    if not input_path.exists():
        raise PipelineError(f"Cleaned input does not exist: {input_path}")
    output_path = root / config["outputs"]["direct_chatml"]
    rejected_path = root / config["outputs"]["direct_rejected"]
    report_path = root / config["outputs"]["report"]
    by_source: Counter[str] = Counter()
    skipped_reverse = 0
    rejected: list[dict[str, Any]] = []

    def records() -> Iterator[dict[str, Any]]:
        nonlocal skipped_reverse
        for record in read_jsonl(input_path):
            mode = record.get("metadata", {}).get("normalization")
            if mode != "direct_pair":
                skipped_reverse += 1
                continue
            try:
                chat = direct_pair_to_chatml(record, str(config["system_prompt"]))
            except (TypeError, ValueError) as exc:
                rejected.append({
                    "source": record.get("source"),
                    "source_record_id": record.get("source_record_id"),
                    "reason": str(exc),
                })
                continue
            by_source[str(record["source"])] += 1
            yield chat

    accepted, accepted_hash = atomic_write_jsonl(output_path, records())
    rejected_count, rejected_hash = atomic_write_jsonl(rejected_path, rejected)
    section = {
        "accepted": accepted,
        "rejected": rejected_count,
        "skipped_reverse_instruction": skipped_reverse,
        "by_source": dict(sorted(by_source.items())),
        "output": {"path": config["outputs"]["direct_chatml"], "sha256": accepted_hash},
        "rejected_output": {"path": config["outputs"]["direct_rejected"], "sha256": rejected_hash},
    }
    report = _load_report(report_path)
    report["version"] = int(config["version"])
    report["direct_pair"] = section
    atomic_write_json(report_path, report)
    return section


def _priority(seed: int, record: dict[str, Any]) -> int:
    identifier = record.get("source_record_id") or record.get("url") or record.get("raw_content", "")
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def prepare_reverse_candidates(
    *, config_path: Path = Path("config/normalization.yaml"), root: Path = Path(".")
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    input_path = root / config["input"]
    output_path = root / config["outputs"]["reverse_candidates"]
    report_path = root / config["outputs"]["report"]
    size = int(config["reverse_instruction"]["sample_size"])
    seed = int(config["seed"])
    pools: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    eligible = 0

    for record in read_jsonl(input_path):
        if record.get("metadata", {}).get("normalization") != "reverse_instruction":
            continue
        eligible += 1
        key = (str(record.get("source")), str(record.get("category")))
        priority = _priority(seed, record)
        identifier = str(record.get("source_record_id") or record.get("url"))
        item = (-priority, identifier, record)
        pool = pools[key]
        if len(pool) < size:
            heapq.heappush(pool, item)
        elif item > pool[0]:
            heapq.heapreplace(pool, item)

    ordered: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, pool in pools.items():
        ordered[key] = [item[2] for item in sorted(pool, key=lambda item: (-item[0], item[1]))]
    by_source_pool: dict[str, list[dict[str, Any]]] = {}
    for source in sorted({key[0] for key in ordered}):
        source_keys = sorted(key for key in ordered if key[0] == source)
        source_values: list[dict[str, Any]] = []
        offset = 0
        while len(source_values) < size:
            added = False
            for key in source_keys:
                values = ordered[key]
                if offset < len(values):
                    source_values.append(values[offset])
                    added = True
            if not added:
                break
            offset += 1
        by_source_pool[source] = source_values

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < size:
        added = False
        for source in sorted(by_source_pool):
            values = by_source_pool[source]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
        offset += 1

    sample = []
    by_source: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for record in selected:
        source = str(record["source"])
        category = str(record["category"])
        by_source[source] += 1
        by_category[category] += 1
        sample.append({
            "candidate_id": record.get("source_record_id"),
            "source": source,
            "category": category,
            "license": record.get("license"),
            "url": record.get("url"),
            "attribution": record.get("attribution"),
            "assistant_reference": record.get("raw_content"),
        })
    count, output_hash = atomic_write_jsonl(output_path, sample)
    section = {
        "eligible": eligible,
        "sample_size": count,
        "by_source": dict(sorted(by_source.items())),
        "by_category": dict(sorted(by_category.items())),
        "output": {"path": config["outputs"]["reverse_candidates"], "sha256": output_hash},
        "generation_status": "awaiting_api_configuration",
        "review_status": "pending",
    }
    report = _load_report(report_path)
    report["version"] = int(config["version"])
    report["reverse_instruction"] = section
    atomic_write_json(report_path, report)
    return section
