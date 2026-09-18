from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return int(ordered[index])


def _summary(values: list[int], limit: int) -> dict[str, Any]:
    over = [value for value in values if value > limit]
    return {
        "records": len(values),
        "minimum": min(values) if values else None,
        "mean": round(statistics.fmean(values), 2) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values) if values else None,
        "over_limit": len(over),
        "over_limit_rate": round(len(over) / len(values), 8) if values else None,
    }


def _batches(values: Iterable[tuple[list[dict[str, str]], str]], size: int):
    batch: list[tuple[list[dict[str, str]], str]] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            messages = value.get("messages")
            category = str(value.get("meta", {}).get("category", "unknown"))
            if not isinstance(messages, list):
                raise ValueError(f"Invalid messages at {path}:{line_number}")
            yield messages, category


def audit(path: Path, tokenizer: Any, *, limit: int, batch_size: int) -> dict[str, Any]:
    lengths: list[int] = []
    by_category: dict[str, list[int]] = defaultdict(list)
    for batch in _batches(_records(path), batch_size):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages, _ in batch
        ]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
        batch_lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
        lengths.extend(batch_lengths)
        for (_, category), length in zip(batch, batch_lengths):
            by_category[category].append(length)
    return {
        "path": path.as_posix(),
        "overall": _summary(lengths, limit),
        "by_category": {
            category: _summary(values, limit)
            for category, values in sorted(by_category.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit rendered ChatML token lengths")
    parser.add_argument("--model", default="Modelo/Qwen3.5-4B")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="JSONL split; repeat for multiple files",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.max_length <= 0 or args.batch_size <= 0:
        parser.error("--max-length and --batch-size must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = args.input or [
        "data/processed/train.jsonl",
        "data/processed/val.jsonl",
        "data/processed/test.jsonl",
    ]
    result = {
        "model": args.model,
        "max_length": args.max_length,
        "splits": [
            audit(Path(value), tokenizer, limit=args.max_length, batch_size=args.batch_size)
            for value in inputs
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
