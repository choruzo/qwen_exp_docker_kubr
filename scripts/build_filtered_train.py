#!/usr/bin/env python3
"""Derive a train split without reverse-instruction records of audit tier A.

Tier A marks reverse-instruction pairs whose answer teaches open-ended enumeration:
changelog sections, narrow questions answered with lists, or long markdown lists
(see docs/16-auditoria-reverse-instruction-v1.md). The frozen ``train.jsonl`` is verified
against ``split_statistics.json`` and never modified; kept lines are copied byte for byte so
the runner can rebuild and verify the derivation from the parent and the excluded hashes.

Outputs (under ``--output-dir``):
- ``<name>.jsonl``: derived train split (large, gitignored, reproducible).
- ``<name>.excluded_hashes.txt``: sorted content hashes removed, one per line.
- ``<name>.manifest.json``: parent/criterion/output identity and aggregate counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "scripts" / "audit_reverse_instruction.py"
TIER_A_FLAGS = ("long_list", "narrow_question_list_answer", "release_notes")
THRESHOLDS = {
    "bullets": 10,
    "length_ratio": 15.0,
    "term_recall": 0.3,
    "versions": 10,
    "narrow_list_bullets": 5,
}


def _load_audit():
    spec = importlib.util.spec_from_file_location("audit_reverse_instruction", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_tier_a(record: dict, audit) -> bool:
    if record.get("meta", {}).get("normalization") != "reverse_instruction":
        return False
    return bool(set(audit.flags(audit.measure(record), THRESHOLDS)) & set(TIER_A_FLAGS))


def build(train: Path, statistics: Path, output_dir: Path, name: str) -> dict:
    audit = _load_audit()
    recorded = json.loads(statistics.read_text(encoding="utf-8"))["outputs"]["train"]
    parent_sha256 = _sha256(train)
    if parent_sha256 != recorded["sha256"]:
        raise SystemExit("train.jsonl does not match split_statistics.json; refusing to derive")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{name}.jsonl"
    excluded_path = output_dir / f"{name}.excluded_hashes.txt"
    manifest_path = output_dir / f"{name}.manifest.json"

    excluded: set[str] = set()
    excluded_groups: Counter = Counter()
    kept_categories: Counter = Counter()
    parent_count = kept = 0
    temporary = output.with_suffix(".jsonl.tmp")
    with train.open("rb") as source, temporary.open("wb") as sink:
        for raw in source:
            if not raw.strip():
                continue
            parent_count += 1
            record = json.loads(raw)
            meta = record["meta"]
            if is_tier_a(record, audit):
                excluded.add(str(meta["content_hash"]))
                excluded_groups[(meta["source"], meta["category"])] += 1
                continue
            sink.write(raw)
            kept += 1
            kept_categories[meta["category"]] += 1
    if parent_count != recorded["count"]:
        temporary.unlink()
        raise SystemExit("Parent record count differs from split_statistics.json")
    temporary.replace(output)
    excluded_path.write_text("".join(f"{value}\n" for value in sorted(excluded)), encoding="utf-8", newline="\n")

    manifest = {
        "version": 1,
        "name": name,
        "parent": {
            "path": train.as_posix(),
            "sha256": parent_sha256,
            "bytes": train.stat().st_size,
            "count": parent_count,
        },
        "criterion": {
            "normalization": "reverse_instruction",
            "any_flag": list(TIER_A_FLAGS),
            "thresholds": THRESHOLDS,
            "audit_script_sha256": _sha256(AUDIT_PATH),
            "reference": "docs/16-auditoria-reverse-instruction-v1.md",
        },
        "excluded": {
            "path": excluded_path.as_posix(),
            "sha256": _sha256(excluded_path),
            "records": parent_count - kept,
            "unique_hashes": len(excluded),
            "by_source_category": [
                {"source": source, "category": category, "records": count}
                for (source, category), count in sorted(excluded_groups.items())
            ],
        },
        "output": {
            "path": output.as_posix(),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
            "count": kept,
            "by_category": dict(sorted(kept_categories.items())),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--statistics", type=Path, default=Path("data/processed/split_statistics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--name", default="train.ri_tier_a_filtered.v1")
    args = parser.parse_args()
    manifest = build(args.train, args.statistics, args.output_dir, args.name)
    print(json.dumps({
        "output": manifest["output"]["path"],
        "count": manifest["output"]["count"],
        "sha256": manifest["output"]["sha256"],
        "excluded_records": manifest["excluded"]["records"],
    }, indent=2))


if __name__ == "__main__":
    main()
