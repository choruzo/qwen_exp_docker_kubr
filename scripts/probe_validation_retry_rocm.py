#!/usr/bin/env python3
"""Re-run only truncated primary generations with a validation-only retry policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docker_k8s_finetune.benchmark.backends import build_backend
from docker_k8s_finetune.benchmark.core import prompt_messages
from docker_k8s_finetune.benchmark.pipeline import _generation_payload
from docker_k8s_finetune.config import load_yaml
from docker_k8s_finetune.io import atomic_write_json, file_sha256, read_jsonl


def run(root: Path, config_path: Path, *, plan: bool = False) -> dict:
    config = load_yaml(config_path)
    if config.get("version") != 1:
        raise ValueError("Unsupported selective retry probe contract")
    source_path = root / config["source_probe"]
    validation_path = root / config["validation"]
    benchmark_path = root / config["frozen_benchmark_config"]
    if file_sha256(source_path) != config["source_probe_sha256"]:
        raise ValueError("Source probe has changed")
    if file_sha256(validation_path) != config["validation_sha256"]:
        raise ValueError("Validation split has changed")
    if file_sha256(benchmark_path) != config["frozen_benchmark_sha256"]:
        raise ValueError("Frozen benchmark config has changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    benchmark = load_yaml(benchmark_path)
    records_by_hash = {
        record["meta"]["content_hash"]: record
        for record in read_jsonl(validation_path)
    }
    selected_hashes = list(source["selected_hashes"])
    if len(selected_hashes) != len(set(selected_hashes)) or any(
        content_hash not in records_by_hash for content_hash in selected_hashes
    ):
        raise ValueError("Source probe selection does not match validation")
    generation = dict(benchmark["generation"])
    generation["retry_on_truncation"] = {
        **generation["retry_on_truncation"],
        "repetition_penalty": float(config["retry_repetition_penalty"]),
    }
    identity = {
        "version": 1,
        "purpose": "validation_only_selective_retry_v1",
        "source_probe_sha256": config["source_probe_sha256"],
        "validation_sha256": config["validation_sha256"],
        "frozen_benchmark_sha256": config["frozen_benchmark_sha256"],
        "selected_hashes": selected_hashes,
        "generation": generation,
    }
    retry_indexes = [
        index for index, record in enumerate(source["records"])
        if int(record.get("retry_count") or 0) > 0
    ]
    if plan:
        return {**identity, "count": len(selected_hashes), "selective_retries": len(retry_indexes)}

    output_path = root / config["output"]
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing selective retry results have a different contract")
    else:
        payload = {**identity, "records": []}
    if [record.get("content_hash") for record in payload["records"]] != selected_hashes[:len(payload["records"])]:
        raise ValueError("Existing selective retry records differ from the source probe")

    backend = None
    max_new_tokens = int(generation["max_new_tokens"])
    for index in range(len(payload["records"]), len(selected_hashes)):
        source_record = source["records"][index]
        content_hash = selected_hashes[index]
        if int(source_record.get("retry_count") or 0) == 0:
            result_record = dict(source_record)
            result_record["selective_retry_applied"] = False
        else:
            if backend is None:
                backend = build_backend(config["variant"], root)
            validation_record = records_by_hash[content_hash]
            retry_generation = dict(generation)
            retry_generation["repetition_penalty"] = float(config["retry_repetition_penalty"])
            generated = _generation_payload(backend.generate(
                prompt_messages(validation_record, generation["response_instruction"]),
                retry_generation,
            ))
            completion_tokens = generated["completion_tokens"]
            result_record = {
                "content_hash": content_hash,
                "category": source_record["category"],
                "reference": source_record["reference"],
                "prediction": generated["prediction"],
                "completion_tokens": completion_tokens,
                "finished_eos": generated["finished_eos"],
                "retry_count": 1,
                "generation_batch_size": source_record["generation_batch_size"],
                "generated_completion_tokens_total": max_new_tokens + int(completion_tokens or 0),
                "truncated": bool(
                    completion_tokens is not None
                    and completion_tokens >= max_new_tokens
                    and generated["finished_eos"] is not True
                ),
                "selective_retry_applied": True,
            }
        payload["records"].append(result_record)
        atomic_write_json(output_path, payload)
        print(
            f"selective-retry: {index + 1}/{len(selected_hashes)} "
            f"applied={result_record['selective_retry_applied']} "
            f"tokens={result_record['completion_tokens']} eos={result_record['finished_eos']}",
            flush=True,
        )
    return {
        "output": str(output_path),
        "count": len(payload["records"]),
        "selective_retries": len(retry_indexes),
        "truncated": sum(bool(record["truncated"]) for record in payload["records"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/validation_retry_probe.rocm.best_epoch_v1.yaml"),
    )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run(root, root / args.config, plan=args.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
