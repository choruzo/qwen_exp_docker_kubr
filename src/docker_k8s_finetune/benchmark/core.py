from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import PipelineError
from ..io import file_sha256


def judge_provenance(config: Mapping[str, Any], root: Path) -> dict[str, Any] | None:
    judge_config = config.get("llm_judge", {})
    if judge_config.get("enabled") is False:
        return None
    manifest_path = root / str(config["inputs"]["judge_manifest"])
    prompt_path = root / str(judge_config["prompt"])
    for path, label in ((manifest_path, "judge manifest"), (prompt_path, "judge prompt")):
        if not path.is_file():
            raise PipelineError(f"Missing {label}: {path}")
    provenance = {
        "manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "prompt": {
            "path": prompt_path.relative_to(root).as_posix(),
            "sha256": file_sha256(prompt_path),
        },
    }
    for config_key, provenance_key in (
        ("test_manifest", "test_manifest"),
        ("out_of_domain", "out_of_domain"),
    ):
        configured = config.get("inputs", {}).get(config_key)
        if configured is None:
            continue
        path = root / str(configured)
        if not path.is_file():
            raise PipelineError(f"Missing benchmark {config_key}: {path}")
        provenance[provenance_key] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
        }
    return provenance


def syntax_provenance(config: Mapping[str, Any], root: Path) -> dict[str, Any] | None:
    syntax_config = config.get("syntax_validation", {})
    if syntax_config.get("enabled") is not True:
        return None
    configured = syntax_config.get("config")
    if not configured:
        raise PipelineError("Enabled benchmark syntax validation requires a config path")
    path = root / str(configured)
    if not path.is_file():
        raise PipelineError(f"Missing benchmark syntax config: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def role_content(record: Mapping[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if isinstance(message, Mapping) and message.get("role") == role:
            return str(message.get("content", "")).strip()
    raise PipelineError(f"Benchmark record has no {role} message")


def prompt_messages(
    record: Mapping[str, Any], response_instruction: str = ""
) -> list[dict[str, str]]:
    messages = []
    for message in record.get("messages", []):
        if not isinstance(message, Mapping) or message.get("role") == "assistant":
            continue
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if role in {"system", "user"} and content:
            messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise PipelineError("Benchmark prompt must end in a user message")
    instruction = response_instruction.strip()
    if instruction:
        if messages[0]["role"] == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    return messages


def canonical_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    fence = re.fullmatch(r"\`\`\`(?:bash|sh|shell|console)?\s*\n?(.*?)\n?\`\`\`", normalized, re.DOTALL | re.IGNORECASE)
    if fence:
        normalized = fence.group(1)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\s*\n\s*", "\n", normalized)
    return normalized.strip()


def exact_match(reference: str, prediction: str) -> float:
    return float(canonical_answer(reference) == canonical_answer(prediction))


def parse_judge_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        correctness = float(payload["correctness"])
        completeness = float(payload["completeness"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Judge response lacks numeric scores: {payload}") from exc
    if not (1.0 <= correctness <= 5.0 and 1.0 <= completeness <= 5.0):
        raise PipelineError(f"Judge scores must be in [1, 5]: {payload}")
    return {
        "correctness": correctness,
        "completeness": completeness,
        "mean": (correctness + completeness) / 2.0,
        "rationale": str(payload.get("rationale", "")).strip(),
    }


def mean_ci(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return {"count": 0, "mean": None, "stddev": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(data)
    stddev = statistics.stdev(data) if len(data) > 1 else 0.0
    margin = 1.96 * stddev / math.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "count": len(data),
        "mean": round(mean, 8),
        "stddev": round(stddev, 8),
        "ci95_low": round(mean - margin, 8),
        "ci95_high": round(mean + margin, 8),
    }


def aggregate_results(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        buckets[str(record["category"])].append(record)

    def summarize(values: list[Mapping[str, Any]]) -> dict[str, Any]:
        generated = [value for value in values if not value.get("error")]
        exact_values = [float(value["exact_match"]) for value in generated if value.get("exact_match") is not None]
        semantic_values = [float(value["semantic_similarity"]) for value in generated if value.get("semantic_similarity") is not None]
        syntax_values = [float(value["syntax_valid"]) for value in generated if value.get("syntax_valid") is not None]
        judge_values = [float(value["judge"]["mean"]) for value in generated if isinstance(value.get("judge"), Mapping)]
        truncated_values = [
            float(bool(value["truncated"]))
            for value in generated
            if value.get("truncated") is not None
        ]
        retry_values = [float(int(value.get("retry_count") or 0) > 0) for value in generated]
        attempt_values = [float(value.get("generation_attempts") or 1) for value in generated]
        generated_token_values = [
            float(value["generated_completion_tokens_total"])
            for value in generated
            if value.get("generated_completion_tokens_total") is not None
        ]
        batch_throughput_values = [
            float(value["batch_tokens_per_second"])
            for value in generated
            if value.get("batch_tokens_per_second") is not None
        ]
        actual_batch_sizes = [
            float(value.get("generation_batch_size") or 1) for value in generated
        ]
        return {
            "count": len(values),
            "generated": len(generated),
            "errors": len(values) - len(generated),
            "exact_match": mean_ci(exact_values),
            "semantic_similarity": mean_ci(semantic_values),
            "syntax_validity": mean_ci(syntax_values),
            "llm_judge": mean_ci(judge_values),
            "truncation_rate": mean_ci(truncated_values),
            "retry_rate": mean_ci(retry_values),
            "generation_attempts": mean_ci(attempt_values),
            "generated_completion_tokens_total": mean_ci(generated_token_values),
            "latency_seconds": mean_ci(float(value["latency_seconds"]) for value in generated),
            "tokens_per_second": mean_ci(
                float(value["tokens_per_second"]) for value in generated
                if value.get("tokens_per_second") is not None
            ),
            "batch_tokens_per_second": mean_ci(batch_throughput_values),
            "generation_batch_size": mean_ci(actual_batch_sizes),
        }

    return {
        "overall": summarize(all_records),
        "by_category": {category: summarize(values) for category, values in sorted(buckets.items())},
    }
