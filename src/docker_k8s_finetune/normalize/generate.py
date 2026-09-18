from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Iterator, Mapping, TypeVar

import requests

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, content_hash, read_jsonl, stable_json
from .core import reverse_pair_to_chatml

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")


class GeneratedInstructionRejected(PipelineError):
    """The model answered, but every retry failed deterministic quality checks."""

_NON_SELF_CONTAINED = re.compile(
    r"(?i)(?:"
    r"\b(?:fragmento|referencia|contenido|documento|documentación|informe|enlace|ejemplo|datos)"
    r"(?:\s+(?:de\s+)?[\wáéíóúñ.-]+){0,3}\s+proporcionad[oa]s?\b|"
    r"\bsegún\s+(?:el|la)\s+(?:fragmento|referencia|contenido|documento|documentación|informe|ejemplo)\b|"
    r"\bbas(?:ado|ada|ándote)\s+en\s+(?:el|la)\s+(?:fragmento|referencia|contenido|documento|documentación|informe|ejemplo)\b|"
    r"\bmencionad[oa]\s+en\s+(?:el|la)\s+(?:fragmento|referencia|contenido|documento|documentación|informe|ejemplo)\b|"
    r"^\s*escribe\s+una\s+instrucción\b"
    r")"
)


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
    user = parsed["user"].strip()
    if _NON_SELF_CONTAINED.search(user):
        raise ValueError("generator instruction is not self-contained")
    return user


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
        "max_tokens": int(reverse.get("max_new_tokens", 512)),
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
            request_payload = dict(payload)
            request_payload["seed"] = int(config["seed"]) + attempt
            response = client.post(
                _endpoint(base_url), headers=headers, json=request_payload,
                timeout=float(reverse["timeout_seconds"])
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return parse_generated_user(str(content)), model
        except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    message = f"Reverse-instruction API failed after {attempts} attempts: {last_error}"
    if isinstance(last_error, ValueError):
        raise GeneratedInstructionRejected(message)
    raise PipelineError(message)


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


def _ordered_parallel_map(
    function: Callable[[_T], _R], values: Iterable[_T], *, max_workers: int
) -> Iterator[_R]:
    """Map with bounded concurrency while preserving input/output order."""
    if max_workers <= 1:
        for value in values:
            yield function(value)
        return

    iterator = iter(values)
    pending: deque[Future[_R]] = deque()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reverse-generation") as executor:
        for _ in range(max_workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def _read_generation_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("fingerprint") != fingerprint:
        return None
    return value


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
    cached_rejected = 0
    generated = 0
    max_workers = max(1, int(reverse.get("parallel_workers", 1)))
    cache_locks: dict[str, Lock] = {}
    cache_locks_guard = Lock()

    def process_candidate(
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
        identifier = str(candidate.get("candidate_id") or content_hash(stable_json(candidate)))
        legacy_fingerprint = content_hash(stable_json({
            "candidate": candidate,
            "prompt": reverse["prompt"],
            "model": os.getenv(str(reverse["model_env"]), ""),
            "seed": config["seed"],
        }))
        fingerprint = content_hash(stable_json({
            "candidate": candidate,
            "prompt": reverse["prompt"],
            "model": os.getenv(str(reverse["model_env"]), ""),
            "seed": config["seed"],
            "generation": {
                "temperature": 0.2,
                "max_new_tokens": int(reverse.get("max_new_tokens", 512)),
            },
        }))
        cache_path = cache_dir / f"{identifier}.json"
        with cache_locks_guard:
            cache_lock = cache_locks.setdefault(identifier, Lock())
        try:
            was_cached = False
            with cache_lock:
                cache: dict[str, Any] | None = None
                value = _read_generation_cache(cache_path, fingerprint)
                if value is None:
                    value = _read_generation_cache(cache_path, legacy_fingerprint)
                    if value is not None:
                        value = dict(value)
                        value["fingerprint"] = fingerprint
                        atomic_write_json(cache_path, value)
                if value is not None:
                    cached_reason = value.get("rejected")
                    if isinstance(cached_reason, str) and cached_reason:
                        return None, {
                            "candidate_id": identifier,
                            "source": candidate.get("source"),
                            "reason": cached_reason,
                        }, True
                    cached_user = value.get("user")
                    if isinstance(cached_user, str) and not _NON_SELF_CONTAINED.search(cached_user):
                        cache = value
                        was_cached = True
                if cache is None:
                    with requests.Session() as session:
                        user, model = generate_user(candidate, config, session=session)
                    cache = {"fingerprint": fingerprint, "user": user, "model": model}
                    atomic_write_json(cache_path, cache)
            return (
                reverse_pair_to_chatml(
                    candidate, str(cache["user"]), str(config["system_prompt"]), str(cache["model"])
                ),
                None,
                was_cached,
            )
        except (GeneratedInstructionRejected, ValueError) as exc:
            LOGGER.error("Rejected reverse candidate %s: %s", identifier, exc)
            rejection = {
                "candidate_id": identifier,
                "source": candidate.get("source"),
                "reason": str(exc),
            }
            atomic_write_json(
                cache_path,
                {"fingerprint": fingerprint, "rejected": str(exc)},
            )
            return None, rejection, False
        except (OSError, PipelineError) as exc:
            LOGGER.error("Rejected reverse candidate %s: %s", identifier, exc)
            return None, {
                "candidate_id": identifier,
                "source": candidate.get("source"),
                "reason": str(exc),
            }, False

    def records() -> Iterator[dict[str, Any]]:
        nonlocal cached, cached_rejected, generated
        for record, rejection, was_cached in _ordered_parallel_map(
            process_candidate, candidates, max_workers=max_workers
        ):
            if rejection is not None:
                if was_cached:
                    cached_rejected += 1
                rejected.append(rejection)
                continue
            if was_cached:
                cached += 1
            else:
                generated += 1
            if record is not None:
                yield record

    accepted, output_hash = atomic_write_jsonl(output_path, records())
    rejected_count, rejected_hash = atomic_write_jsonl(rejected_path, rejected)
    return {
        "scope": scope,
        "accepted": accepted,
        "generated": generated,
        "cached": cached,
        "cached_rejected": cached_rejected,
        "rejected": rejected_count,
        "output": {"path": output_path.relative_to(root).as_posix(), "sha256": output_hash},
        "rejected_output": {"path": rejected_path.relative_to(root).as_posix(), "sha256": rejected_hash},
    }
