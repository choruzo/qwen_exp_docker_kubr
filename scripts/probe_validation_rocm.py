#!/usr/bin/env python3
"""Run a small, resumable ROCm generation probe on validation only."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from docker_k8s_finetune.benchmark.backends import build_backend
from docker_k8s_finetune.benchmark.core import prompt_messages, role_content
from docker_k8s_finetune.benchmark.pipeline import _generate_with_truncation_retry
from docker_k8s_finetune.config import REQUIRED_CATEGORIES, load_yaml
from docker_k8s_finetune.io import atomic_write_json, file_sha256, read_jsonl


def select_validation_records(records: list[dict], per_category: int = 2) -> list[dict]:
    """Preselect typical and long references without observing model outputs."""
    if per_category != 2:
        raise ValueError("v1 selection requires exactly two records per category")
    groups: dict[str, list[dict]] = defaultdict(list)
    hashes: set[str] = set()
    for record in records:
        meta = record.get("meta", {})
        category = str(meta.get("category", ""))
        content_hash = str(meta.get("content_hash", ""))
        if category not in REQUIRED_CATEGORIES or not content_hash or content_hash in hashes:
            raise ValueError("Validation has an invalid category or duplicate/missing content hash")
        hashes.add(content_hash)
        groups[category].append(record)
    if set(groups) != REQUIRED_CATEGORIES:
        raise ValueError("Validation does not cover every expected category")
    selected = []
    for category in sorted(groups):
        ordered = sorted(
            groups[category],
            key=lambda record: (len(role_content(record, "assistant")), record["meta"]["content_hash"]),
        )
        if len(ordered) < 2:
            raise ValueError(f"Too few validation records for {category}")
        median_index = (len(ordered) - 1) // 2
        p90_index = math.ceil(0.9 * (len(ordered) - 1))
        if p90_index == median_index:
            p90_index = len(ordered) - 1
        selected.extend((ordered[median_index], ordered[p90_index]))
    return selected


def _load_inputs(root: Path, config: dict) -> tuple[list[dict], dict]:
    frozen_path = root / config["frozen_benchmark_config"]
    validation_path = root / config["validation"]
    if file_sha256(frozen_path) != config["frozen_benchmark_sha256"]:
        raise ValueError("Frozen benchmark config has changed")
    if file_sha256(validation_path) != config["validation_sha256"]:
        raise ValueError("Validation split has changed")
    records = list(read_jsonl(validation_path))
    selected = select_validation_records(records, int(config["sample_per_category"]))
    held_out_hashes = {
        str(record["content_hash"])
        for record in read_jsonl(root / config["test_manifest"])
    }
    held_out_hashes.update(
        str(record["meta"]["content_hash"])
        for record in read_jsonl(root / config["out_of_domain"])
    )
    if any(record["meta"]["content_hash"] in held_out_hashes for record in selected):
        raise ValueError("Validation probe overlaps held-out test or out-of-domain fixtures")
    return selected, load_yaml(frozen_path)


def run(root: Path, config_path: Path, variant: str, policy: str, *, limit: int | None = None, plan: bool = False) -> dict:
    config = load_yaml(config_path)
    if config.get("version") != 1 or config.get("selection") != "reference_length_median_and_p90_by_category_v1":
        raise ValueError("Unsupported validation probe contract")
    selected, frozen = _load_inputs(root, config)
    if variant not in config["variants"] or policy not in config["policies"]:
        raise ValueError("Unknown variant or policy")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    generation = dict(frozen["generation"])
    chosen = config["policies"][policy]
    generation["repetition_penalty"] = float(chosen["repetition_penalty"])
    generation["retry_on_truncation"] = {
        **generation["retry_on_truncation"],
        "repetition_penalty": float(chosen["retry_repetition_penalty"]),
    }
    identity = {
        "version": 1,
        "probe_config_sha256": file_sha256(config_path),
        "validation_sha256": config["validation_sha256"],
        "frozen_benchmark_sha256": config["frozen_benchmark_sha256"],
        "variant": variant,
        "policy": policy,
        "selected_hashes": [record["meta"]["content_hash"] for record in selected],
        "generation": generation,
    }
    if plan:
        return {**identity, "count": len(selected)}
    output_dir = root / config["outputs_dir"]
    suffix = f"_limit{limit}" if limit is not None else ""
    output_path = output_dir / f"{variant}_{policy}{suffix}_results.json"
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing probe results have a different contract")
    else:
        payload = {**identity, "records": []}
    if len(payload["records"]) > len(selected):
        raise ValueError("Existing probe has too many records")
    if [record.get("content_hash") for record in payload["records"]] != identity["selected_hashes"][:len(payload["records"])]:
        raise ValueError("Existing probe records differ from the selected validation sample")
    backend = build_backend(config["variants"][variant], root)
    for index in range(len(payload["records"]), len(selected)):
        record = selected[index]
        result = _generate_with_truncation_retry(
            backend,
            prompt_messages(record, generation["response_instruction"]),
            generation,
        )
        payload["records"].append({
            "content_hash": record["meta"]["content_hash"],
            "category": record["meta"]["category"],
            "reference": role_content(record, "assistant"),
            "prediction": result["prediction"],
            "completion_tokens": result["completion_tokens"],
            "finished_eos": result["finished_eos"],
            "retry_count": result["retry_count"],
            "truncated": bool(result["completion_tokens"] is not None and result["completion_tokens"] >= generation["max_new_tokens"] and result["finished_eos"] is not True),
        })
        atomic_write_json(output_path, payload)
        print(f"{variant}/{policy}: {index + 1}/{len(selected)} {record['meta']['category']} tokens={result['completion_tokens']} eos={result['finished_eos']}", flush=True)
    return {"output": str(output_path), "count": len(payload["records"]), "truncated": sum(record["truncated"] for record in payload["records"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/validation_probe.rocm.v1.yaml"))
    parser.add_argument("--variant", choices=("adapter", "merged"), required=True)
    parser.add_argument("--policy", choices=("frozen", "repetition_control_v1"), default="frozen")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run(root, root / args.config, args.variant, args.policy, limit=args.limit, plan=args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
