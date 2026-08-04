from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, content_hash, read_jsonl, stable_json


def canonical_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    by_role = {message.get("role"): message.get("content") for message in messages if isinstance(message, dict)}
    if not by_role.get("user") or not by_role.get("assistant"):
        raise ValueError("record lacks user or assistant content")

    def normalize(value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n").replace("\r", "\n")
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    return stable_json({"user": normalize(by_role["user"]), "assistant": normalize(by_role["assistant"])})


def run_exact_dedupe(
    *, config_path: Path = Path("config/dedupe.yaml"), root: Path = Path(".")
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    inputs = [root / path for path in config["inputs"]]
    existing = [path for path in inputs if path.exists()]
    missing = [path for path in inputs if not path.exists()]
    if missing and not config.get("allow_missing_inputs", False):
        raise PipelineError(f"Missing dedupe inputs: {', '.join(str(path) for path in missing)}")
    if not existing:
        raise PipelineError("No normalized inputs are available for exact deduplication")

    exact = config["exact"]
    output_path = root / exact["output"]
    duplicates_path = root / exact["duplicates"]
    report_path = root / exact["report"]
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    input_by_source: Counter[str] = Counter()
    kept_by_source: Counter[str] = Counter()
    removed_by_source: Counter[str] = Counter()

    def unique_records() -> Iterator[dict[str, Any]]:
        for path in existing:
            for record in read_jsonl(path):
                source = str(record.get("meta", {}).get("source", "unknown"))
                input_by_source[source] += 1
                fingerprint = content_hash(canonical_content(record))
                original = seen.get(fingerprint)
                if original is not None:
                    removed_by_source[source] += 1
                    duplicates.append({
                        "fingerprint": fingerprint,
                        "source": source,
                        "source_record_id": record.get("meta", {}).get("source_record_id"),
                        "kept_source": original.get("meta", {}).get("source"),
                        "kept_source_record_id": original.get("meta", {}).get("source_record_id"),
                    })
                    continue
                seen[fingerprint] = record
                kept_by_source[source] += 1
                record.setdefault("meta", {})["content_hash"] = fingerprint
                yield record

    kept, output_hash = atomic_write_jsonl(output_path, unique_records())
    removed, duplicates_hash = atomic_write_jsonl(duplicates_path, duplicates)
    report = {
        "version": int(config["version"]),
        "inputs": [path.relative_to(root).as_posix() for path in existing],
        "missing_inputs": [path.relative_to(root).as_posix() for path in missing],
        "counts": {"input": sum(input_by_source.values()), "kept": kept, "removed": removed},
        "input_by_source": dict(sorted(input_by_source.items())),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "output": {"path": exact["output"], "sha256": output_hash},
        "duplicates": {"path": exact["duplicates"], "sha256": duplicates_hash},
    }
    atomic_write_json(report_path, report)
    return report
