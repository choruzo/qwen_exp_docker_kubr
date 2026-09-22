#!/usr/bin/env python3
"""Mechanistic case study only; never use selected test failures to tune a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docker_k8s_finetune.benchmark.backends import build_backend
from docker_k8s_finetune.benchmark.core import prompt_messages, role_content
from docker_k8s_finetune.benchmark.pipeline import _generate_with_truncation_retry
from docker_k8s_finetune.config import load_yaml
from docker_k8s_finetune.io import atomic_write_json, file_sha256, read_jsonl


CATEGORIES = ("comando_cli", "concepto", "troubleshooting")


def run(root: Path, variant: str, *, plan: bool = False) -> dict:
    probe_config_path = root / "config/validation_probe.rocm.v1.yaml"
    probe = load_yaml(probe_config_path)
    frozen_path = root / probe["frozen_benchmark_config"]
    if file_sha256(frozen_path) != probe["frozen_benchmark_sha256"]:
        raise ValueError("Frozen benchmark config changed")
    frozen = load_yaml(frozen_path)
    test_path = root / frozen["inputs"]["test"]
    statistics = json.loads((root / frozen["inputs"]["split_statistics"]).read_text(encoding="utf-8"))
    test_sha = statistics["outputs"]["test"]["sha256"]
    if file_sha256(test_path) != test_sha:
        raise ValueError("Held-out test changed")
    merged_path = root / frozen["outputs"]["finetuned_safetensors"]
    saved = json.loads(merged_path.read_text(encoding="utf-8"))
    if saved["split_test_sha256"] != test_sha or saved["generation"] != frozen["generation"]:
        raise ValueError("Saved merged benchmark does not match frozen contract")
    by_hash = {record["meta"]["content_hash"]: record for record in read_jsonl(test_path)}
    if len(by_hash) != statistics["outputs"]["test"]["count"]:
        raise ValueError("Held-out test contains duplicate or missing records")
    selected = []
    for category in CATEGORIES:
        candidates = sorted(
            (record for record in saved["records"] if record["category"] == category and record.get("truncated") is True),
            key=lambda record: record["content_hash"],
        )
        if not candidates or candidates[0]["content_hash"] not in by_hash:
            raise ValueError(f"No verified truncated case for {category}")
        selected.append(by_hash[candidates[0]["content_hash"]])
    identity = {
        "version": 1,
        "purpose": "mechanistic_case_study_not_for_policy_selection",
        "frozen_benchmark_sha256": probe["frozen_benchmark_sha256"],
        "test_sha256": test_sha,
        "variant": variant,
        "selected_hashes": [record["meta"]["content_hash"] for record in selected],
        "generation": frozen["generation"],
    }
    if variant not in probe["variants"]:
        raise ValueError("Unknown variant")
    if plan:
        return {**identity, "categories": list(CATEGORIES)}
    output = root / "benchmarks/rocm/hipblaslt_patched/failed_test_case_study_v1" / f"{variant}_results.json"
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing case study has a different contract")
    else:
        payload = {**identity, "records": []}
    if [record.get("content_hash") for record in payload["records"]] != identity["selected_hashes"][:len(payload["records"])]:
        raise ValueError("Existing case-study records differ from the selected cases")
    backend = build_backend(probe["variants"][variant], root)
    for index in range(len(payload["records"]), len(selected)):
        record = selected[index]
        result = _generate_with_truncation_retry(
            backend,
            prompt_messages(record, frozen["generation"]["response_instruction"]),
            frozen["generation"],
        )
        payload["records"].append({
            "content_hash": record["meta"]["content_hash"],
            "category": record["meta"]["category"],
            "reference": role_content(record, "assistant"),
            "prediction": result["prediction"],
            "completion_tokens": result["completion_tokens"],
            "finished_eos": result["finished_eos"],
            "retry_count": result["retry_count"],
            "truncated": bool(result["completion_tokens"] is not None and result["completion_tokens"] >= frozen["generation"]["max_new_tokens"] and result["finished_eos"] is not True),
        })
        atomic_write_json(output, payload)
        print(f"{variant}: {index + 1}/{len(selected)} {record['meta']['category']} tokens={result['completion_tokens']} eos={result['finished_eos']}", flush=True)
    return {"output": str(output), "count": len(payload["records"]), "truncated": sum(record["truncated"] for record in payload["records"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("adapter", "merged"), required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run(root, args.variant, plan=args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
