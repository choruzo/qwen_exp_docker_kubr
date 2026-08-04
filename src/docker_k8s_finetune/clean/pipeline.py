from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, content_hash, read_jsonl
from .core import clean_record


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def run_cleaning(
    *,
    config_path: Path = Path("config/cleaning.yaml"),
    input_dir: Path = Path("data/interim/extracted"),
    root: Path = Path("."),
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    inputs = sorted((root / input_dir).glob("*.jsonl"))
    if not inputs:
        raise PipelineError(f"No extracted JSONL files found in {root / input_dir}")

    outputs = config["outputs"]
    accepted_path = root / outputs["accepted"]
    rejected_path = root / outputs["rejected"]
    report_path = root / outputs["report"]
    review_path = root / outputs["pii_review"]

    accepted_by_source: Counter[str] = Counter()
    rejected_by_source: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    redactions_by_type: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    review_limit = int(config["pii"]["review_sample_size"])
    input_counts: Counter[str] = Counter()

    def accepted_records() -> Iterator[dict[str, Any]]:
        for path in inputs:
            for record in read_jsonl(path):
                source = str(record.get("source", path.stem))
                input_counts[source] += 1
                result = clean_record(record, config)
                for kind, count in result.redactions.items():
                    redactions_by_type[kind] += count
                if result.record is None:
                    reason = result.reason or "unknown"
                    rejected_by_source[source] += 1
                    rejected_by_reason[reason] += 1
                    rejected.append({
                        "source": source,
                        "source_record_id": record.get("source_record_id"),
                        "url": record.get("url"),
                        "reason": reason,
                        "pii_redactions": result.redactions,
                    })
                    continue
                accepted_by_source[source] += 1
                if result.redactions and len(review) < review_limit:
                    review.append({
                        "source": source,
                        "source_record_id": result.record.get("source_record_id"),
                        "redaction_types": sorted(result.redactions),
                        "redacted_excerpt": result.record["raw_content"][:500],
                    })
                yield result.record

    accepted_count, accepted_sha256 = atomic_write_jsonl(accepted_path, accepted_records())
    rejected_count, rejected_sha256 = atomic_write_jsonl(rejected_path, rejected)
    review_count, review_sha256 = atomic_write_jsonl(review_path, review)

    input_files = []
    for path in inputs:
        data = path.read_bytes()
        input_files.append({
            "path": _relative(path, root),
            "bytes": len(data),
            "sha256": content_hash(data),
        })
    report: dict[str, Any] = {
        "version": int(config["version"]),
        "seed": int(config["seed"]),
        "created_at": _utc_now(),
        "inputs": input_files,
        "counts": {
            "input": sum(input_counts.values()),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "pii_review_sample": review_count,
        },
        "input_by_source": dict(sorted(input_counts.items())),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "rejected_by_source": dict(sorted(rejected_by_source.items())),
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "pii_redactions_by_type": dict(sorted(redactions_by_type.items())),
        "outputs": {
            "accepted": {"path": _relative(accepted_path, root), "sha256": accepted_sha256},
            "rejected": {"path": _relative(rejected_path, root), "sha256": rejected_sha256},
            "pii_review": {"path": _relative(review_path, root), "sha256": review_sha256},
        },
    }
    atomic_write_json(report_path, report)
    return report
