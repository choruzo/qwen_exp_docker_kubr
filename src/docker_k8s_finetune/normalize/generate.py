from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import requests

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, content_hash, read_jsonl, stable_json
from .core import reverse_pair_to_chatml

LOGGER = logging.getLogger(__name__)


def _require_environment(config: Mapping[str, Any]) -> None:
    reverse = config["reverse_instruction"]
    missing = [
        name for name in (str(reverse["base_url_env"]), str(reverse["model_env"]))
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise PipelineError(f"Missing reverse-instruction environment variables: {', '.join(missing)}")


def parse_generated_user(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("generator did not return valid JSON") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("user"), str) or not parsed["user"].strip():
        raise ValueError("generator JSON must contain a non-empty string field 'user'")
    return parsed["user"].strip()


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def generate_user(
    candidate: Mapping[str, Any], config: Mapping[str, Any], *, session: requests.Session | None = None
) -> tuple[str, str]:
    reverse = config["reverse_instruction"]
    base_url = os.getenv(str(reverse["base_url_env"]), "").strip()
    model = os.getenv(str(reverse["model_env"]), "").strip()
    api_key = os.getenv(str(reverse["api_key_env"]), "").strip()
    if not base_url or not model:
        raise PipelineError(
            f"Set {reverse['base_url_env']} and {reverse['model_env']} before reverse-instruction generation"
        )
    reference = str(candidate["assistant_reference"])
    maximum = int(reverse["max_reference_chars"])
    if len(reference) > maximum:
        raise ValueError(f"assistant reference exceeds {maximum} characters")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0.2,
        "seed": int(config["seed"]),
        "messages": [
            {"role": "system", "content": str(reverse["prompt"])},
            {"role": "user", "content": f"CATEGORY: {candidate['category']}\n\nREFERENCE:\n{reference}"},
        ],
    }
    client = session or requests.Session()
    attempts = int(reverse["max_retries"])
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post(
                _endpoint(base_url), headers=headers, json=payload, timeout=float(reverse["timeout_seconds"])
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return parse_generated_user(str(content)), model
        except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    raise PipelineError(f"Reverse-instruction API failed after {attempts} attempts: {last_error}")


def _candidate_from_cleaned(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record.get("source_record_id"),
        "source": record.get("source"),
        "category": record.get("category"),
        "license": record.get("license"),
        "url": record.get("url"),
        "attribution": record.get("attribution"),
        "assistant_reference": record.get("raw_content"),
    }


def _full_candidates(input_path: Path) -> Iterator[dict[str, Any]]:
    for record in read_jsonl(input_path):
        if record.get("metadata", {}).get("normalization") == "reverse_instruction":
            yield _candidate_from_cleaned(record)


def _verify_full_gate(root: Path, config: Mapping[str, Any]) -> None:
    reverse = config["reverse_instruction"]
    review = load_yaml(root / reverse["review_config"])
    if review.get("status") != "approved":
        raise PipelineError("Full reverse-instruction generation requires status=approved in the review config")
    sample_path = root / reverse["generated_sample"]
    if not sample_path.exists():
        raise PipelineError("The generated sample must exist before full generation")
    actual_hash = content_hash(sample_path.read_bytes())
    if review.get("sample_sha256") != actual_hash:
        raise PipelineError("Review sample_sha256 does not match the generated sample")


def run_reverse_generation(
    *, scope: str, config_path: Path = Path("config/normalization.yaml"), root: Path = Path(".")
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    reverse = config["reverse_instruction"]
    _require_environment(config)
    if scope == "sample":
        candidates: Iterable[dict[str, Any]] = read_jsonl(root / config["outputs"]["reverse_candidates"])
        output_path = root / reverse["generated_sample"]
        rejected_path = root / reverse["rejected_sample"]
    elif scope == "full":
        _verify_full_gate(root, config)
        candidates = _full_candidates(root / config["input"])
        output_path = root / reverse["generated_full"]
        rejected_path = root / reverse["rejected_full"]
    else:
        raise ValueError("scope must be 'sample' or 'full'")

    cache_dir = root / reverse["cache_dir"]
    rejected: list[dict[str, Any]] = []
    cached = 0
    generated = 0

    def records() -> Iterator[dict[str, Any]]:
        nonlocal cached, generated
        for candidate in candidates:
            identifier = str(candidate.get("candidate_id") or content_hash(stable_json(candidate)))
            fingerprint = content_hash(stable_json({
                "candidate": candidate,
                "prompt": reverse["prompt"],
                "model": os.getenv(str(reverse["model_env"]), ""),
                "seed": config["seed"],
            }))
            cache_path = cache_dir / f"{identifier}.json"
            cache: dict[str, Any] | None = None
            if cache_path.exists():
                with cache_path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                if isinstance(value, dict) and value.get("fingerprint") == fingerprint:
                    cache = value
            try:
                if cache is None:
                    user, model = generate_user(candidate, config)
                    cache = {"fingerprint": fingerprint, "user": user, "model": model}
                    atomic_write_json(cache_path, cache)
                    generated += 1
                else:
                    cached += 1
                yield reverse_pair_to_chatml(
                    candidate, str(cache["user"]), str(config["system_prompt"]), str(cache["model"])
                )
            except (OSError, PipelineError, ValueError) as exc:
                LOGGER.error("Rejected reverse candidate %s: %s", identifier, exc)
                rejected.append({"candidate_id": identifier, "source": candidate.get("source"), "reason": str(exc)})

    accepted, output_hash = atomic_write_jsonl(output_path, records())
    rejected_count, rejected_hash = atomic_write_jsonl(rejected_path, rejected)
    return {
        "scope": scope,
        "accepted": accepted,
        "generated": generated,
        "cached": cached,
        "rejected": rejected_count,
        "output": {"path": output_path.relative_to(root).as_posix(), "sha256": output_hash},
        "rejected_output": {"path": rejected_path.relative_to(root).as_posix(), "sha256": rejected_hash},
    }
