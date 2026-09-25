#!/usr/bin/env python3
"""Report aggregate duplication and repeated n-grams without emitting dataset text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def assistant_text(record: dict) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in record.get("messages", [])
        if message.get("role") == "assistant"
    )


def repeated_ngram_ratio(text: str, size: int) -> float:
    tokens = TOKEN_RE.findall(text.casefold())
    if len(tokens) < size:
        return 0.0
    ngrams = [tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def audit(path: Path, *, ngram_size: int, repetition_threshold: float) -> dict:
    exact = Counter()
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"records": 0, "high_repetition": 0, "long_responses": 0}
    )
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = assistant_text(record)
                meta = record["meta"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc
            records += 1
            exact[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
            key = (str(meta.get("source", "unknown")), str(meta.get("category", "unknown")))
            group = groups[key]
            group["records"] += 1
            group["high_repetition"] += repeated_ngram_ratio(text, ngram_size) >= repetition_threshold
            group["long_responses"] += len(text.split()) >= 1000

    duplicate_groups = [count for count in exact.values() if count > 1]
    return {
        "version": 1,
        "input": str(path),
        "records": records,
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_excess_records": sum(count - 1 for count in duplicate_groups),
        "maximum_exact_frequency": max(duplicate_groups, default=1),
        "ngram_size": ngram_size,
        "repetition_threshold": repetition_threshold,
        "by_source_category": [
            {
                "source": source,
                "category": category,
                **values,
                "high_repetition_rate": values["high_repetition"] / values["records"],
                "long_response_rate": values["long_responses"] / values["records"],
            }
            for (source, category), values in sorted(groups.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/processed/train.jsonl"))
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--repetition-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.ngram_size < 1 or not 0 <= args.repetition_threshold <= 1:
        parser.error("ngram size must be positive and threshold must be between 0 and 1")
    print(json.dumps(
        audit(args.path, ngram_size=args.ngram_size, repetition_threshold=args.repetition_threshold),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
