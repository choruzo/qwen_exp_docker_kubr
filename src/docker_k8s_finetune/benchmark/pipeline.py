from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, content_hash, file_sha256, read_jsonl, stable_json
from ..schema import utc_now_iso
from .backends import build_backend
from .core import (
    aggregate_results,
    exact_match,
    judge_provenance,
    parse_judge_payload,
    prompt_messages,
    role_content,
    syntax_provenance,
)
from .scoring import JudgeClient, semantic_scores, syntax_scores


LOGGER = logging.getLogger(__name__)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineError(f"Frozen benchmark manifest does not exist: {path}")
    values = list(read_jsonl(path))
    if not values:
        raise PipelineError(f"Frozen benchmark manifest is empty: {path}")
    return values


def _validated_judge_hashes(
    config: dict[str, Any],
    root: Path,
    expected_records: list[dict[str, Any]],
    split_statistics: dict[str, Any],
) -> set[str]:
    manifest = _load_manifest(root / str(config["inputs"]["judge_manifest"]))
    hashes = [str(item.get("content_hash", "")) for item in manifest]
    expected_count = int(split_statistics.get("benchmark_manifests", {}).get("judge", 0))
    test_count = int(split_statistics.get("outputs", {}).get("test", {}).get("count", 0))
    if expected_count <= 0 or len(manifest) != expected_count:
        raise PipelineError(
            f"Judge manifest count differs from split statistics: {len(manifest)} != {expected_count}"
        )
    if any(not value for value in hashes) or len(set(hashes)) != len(hashes):
        raise PipelineError("Judge manifest content hashes must be non-empty and unique")
    if test_count <= 0 or test_count > len(expected_records):
        raise PipelineError("Frozen test count is inconsistent with benchmark records")
    test_by_hash = {
        str(record.get("meta", {}).get("content_hash")): record
        for record in expected_records[:test_count]
    }
    unknown = sorted(set(hashes) - set(test_by_hash))
    if unknown:
        raise PipelineError("Judge manifest contains records outside the frozen held-out test")
    for item in manifest:
        record_meta = test_by_hash[str(item["content_hash"])].get("meta", {})
        for field in ("category", "source"):
            if item.get(field) is not None and record_meta.get(field) is not None:
                if str(item[field]) != str(record_meta[field]):
                    raise PipelineError(f"Judge manifest {field} differs from frozen test metadata")
    return set(hashes)


def _load_benchmark_records(
    config: dict[str, Any], root: Path, limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_path = root / str(config["inputs"]["test"])
    if not test_path.is_file():
        raise PipelineError(f"Held-out test set does not exist: {test_path}")
    statistics_path = root / str(config["inputs"]["split_statistics"])
    if not statistics_path.is_file():
        raise PipelineError(f"Split statistics do not exist: {statistics_path}")
    try:
        split_statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read split statistics: {exc}") from exc
    expected_test_hash = split_statistics.get("outputs", {}).get("test", {}).get("sha256")
    if not expected_test_hash or file_sha256(test_path) != expected_test_hash:
        raise PipelineError("Held-out test set does not match split_statistics.json")
    by_hash = {str(record["meta"]["content_hash"]): record for record in read_jsonl(test_path)}
    manifest = _load_manifest(root / str(config["inputs"]["test_manifest"]))
    manifest_hashes = [str(item["content_hash"]) for item in manifest]
    if len(manifest_hashes) != len(set(manifest_hashes)):
        raise PipelineError("Frozen test manifest content hashes must be unique")
    if len(manifest_hashes) != len(by_hash) or set(manifest_hashes) != set(by_hash):
        raise PipelineError("Frozen test manifest does not exactly match data/processed/test.jsonl")
    selected = [by_hash[value] for value in manifest_hashes]
    if limit is not None:
        selected = selected[:limit]
    ood_path = root / str(config["inputs"]["out_of_domain"])
    ood = list(read_jsonl(ood_path)) if ood_path.is_file() else []
    if not ood:
        raise PipelineError(f"Out-of-domain fixture is missing or empty: {ood_path}")
    expected_ood_count = int(config["inputs"].get("out_of_domain_count", 0))
    if expected_ood_count <= 0 or len(ood) != expected_ood_count:
        raise PipelineError(
            f"Out-of-domain fixture count mismatch: {len(ood)} != {expected_ood_count}"
        )
    ood_hashes = [str(record.get("meta", {}).get("content_hash", "")) for record in ood]
    if any(not value for value in ood_hashes) or len(set(ood_hashes)) != len(ood_hashes):
        raise PipelineError("Out-of-domain content hashes must be non-empty and unique")
    if set(ood_hashes) & set(by_hash):
        raise PipelineError("Out-of-domain fixture overlaps the frozen held-out test")
    for record in ood:
        meta = record.get("meta", {})
        if meta.get("category") != "out_of_domain":
            raise PipelineError("Out-of-domain fixture contains a non-OOD category")
        if any(not str(meta.get(field, "")).strip() for field in ("source", "license", "url")):
            raise PipelineError("Out-of-domain fixture lacks source, license or URL provenance")
        role_content(record, "user")
        role_content(record, "assistant")
    return selected + ood, split_statistics


def _output_path(config: dict[str, Any], variant: str, root: Path) -> Path:
    try:
        return root / str(config["outputs"][variant])
    except KeyError as exc:
        raise PipelineError(f"Unknown benchmark variant: {variant}") from exc


def _update_combined(config: dict[str, Any], root: Path) -> None:
    output = root / str(config["outputs"]["finetuned_combined"])
    values: dict[str, Any] = {"version": 1, "created_at": utc_now_iso(), "variants": {}}
    for variant in ("finetuned_safetensors", "finetuned_gguf"):
        path = _output_path(config, variant, root)
        if path.is_file():
            values["variants"][variant] = json.loads(path.read_text(encoding="utf-8"))
    atomic_write_json(output, values)


def _is_provisional(split_statistics: dict[str, Any], completion: dict[str, bool]) -> bool:
    return bool(split_statistics.get("provisional") or not all(completion.values()))


def _generation_cache_fingerprint(
    record: dict[str, Any], *, variant: str, backend: dict[str, Any], generation: dict[str, Any],
) -> str:
    generation_request = {
        key: value
        for key, value in generation.items()
        if key not in {
            "max_truncation_rate",
            "cache_compatible_prior_max_new_tokens",
        }
    }
    return content_hash(stable_json({
        "variant": variant,
        "backend": backend,
        "generation": generation_request,
        "content_hash": record.get("meta", {}).get("content_hash"),
        "messages": record.get("messages"),
    }))


def _compatible_cached_generation(
    candidate: dict[str, Any],
    record: dict[str, Any],
    *,
    variant: str,
    backend: dict[str, Any],
    generation: dict[str, Any],
    current_fingerprint: str,
) -> tuple[dict[str, Any] | None, int | None]:
    cached_generation = candidate.get("generation")
    if not isinstance(cached_generation, dict):
        return None, None
    if candidate.get("fingerprint") == current_fingerprint:
        return cached_generation, None
    completion_tokens = cached_generation.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        return None, None
    for prior_limit in generation.get("cache_compatible_prior_max_new_tokens", []):
        prior_generation = dict(generation)
        prior_generation["max_new_tokens"] = int(prior_limit)
        prior_fingerprint = _generation_cache_fingerprint(
            record,
            variant=variant,
            backend=backend,
            generation=prior_generation,
        )
        if (
            candidate.get("fingerprint") == prior_fingerprint
            and completion_tokens < int(prior_limit)
        ):
            return cached_generation, int(prior_limit)
    return None, None


def _normalized_generation(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    completion_tokens = normalized.get("completion_tokens")
    normalized.setdefault("generation_attempts", 1)
    normalized.setdefault("retry_count", 0)
    normalized.setdefault("generated_completion_tokens_total", completion_tokens)
    normalized.setdefault("generation_batch_size", 1)
    normalized.setdefault("finished_eos", None)
    return normalized


def _legacy_retry_policy_cache(
    candidate: dict[str, Any],
    record: dict[str, Any],
    *,
    variant: str,
    backend: dict[str, Any],
    generation: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    policy = generation.get("retry_on_truncation")
    cached = candidate.get("generation")
    if not isinstance(policy, dict) or policy.get("enabled") is not True or not isinstance(cached, dict):
        return None, None
    completion_tokens = cached.get("completion_tokens")
    if not isinstance(completion_tokens, int):
        return None, None
    legacy = dict(generation)
    legacy.pop("retry_on_truncation", None)
    limits = [int(generation["max_new_tokens"]), *generation.get("cache_compatible_prior_max_new_tokens", [])]
    for limit in limits:
        prior = dict(legacy)
        prior["max_new_tokens"] = int(limit)
        fingerprint = _generation_cache_fingerprint(
            record,
            variant=variant,
            backend=backend,
            generation=prior,
        )
        if candidate.get("fingerprint") != fingerprint:
            continue
        if completion_tokens < int(limit):
            return _normalized_generation(cached), "reuse"
        if int(limit) == int(generation["max_new_tokens"]):
            return _normalized_generation(cached), "retry"
        return None, None
    return None, None


def _generation_payload(generated: Any) -> dict[str, Any]:
    payload = {
        "prediction": generated.text,
        "latency_seconds": generated.latency_seconds,
        "prompt_tokens": generated.prompt_tokens,
        "completion_tokens": generated.completion_tokens,
        "tokens_per_second": generated.tokens_per_second,
        "generation_attempts": 1,
        "retry_count": 0,
        "generated_completion_tokens_total": generated.completion_tokens,
        "generation_batch_size": int(getattr(generated, "batch_size", 1)),
        "finished_eos": getattr(generated, "finished_eos", None),
    }
    batch_size = int(payload["generation_batch_size"])
    if batch_size > 1:
        payload["batch_completion_tokens_total"] = getattr(
            generated, "batch_completion_tokens_total", None
        )
        payload["batch_tokens_per_second"] = getattr(
            generated, "batch_tokens_per_second", None
        )
    return payload


def _backend_generate_batch(
    backend: Any,
    messages_batch: list[list[dict[str, str]]],
    generation: dict[str, Any],
) -> list[Any]:
    generate_batch = getattr(backend, "generate_batch", None)
    if callable(generate_batch):
        generated = list(generate_batch(messages_batch, generation))
    else:
        generated = [backend.generate(messages, generation) for messages in messages_batch]
    if len(generated) != len(messages_batch):
        raise PipelineError(
            f"Backend returned {len(generated)} generations for a batch of {len(messages_batch)}"
        )
    return generated


def _prefill_generation_cache(
    *,
    backend: Any,
    input_records: list[dict[str, Any]],
    variant: str,
    backend_config: dict[str, Any],
    generation: dict[str, Any],
    cache_dir: Path,
) -> set[str]:
    """Fill deterministic batch groups before result aggregation.

    A group is reused only when every member has the current fingerprint and
    the same derived group hash. A power loss between per-record atomic writes
    therefore causes at most that complete group to be regenerated, without
    shifting the remaining records into different batches.
    """
    batch_size = int(generation.get("batch_size", 1))
    if batch_size <= 1:
        return set()
    planned: list[
        tuple[str, Path, str, list[dict[str, str]], int, dict[str, Any] | None]
    ] = []
    response_instruction = str(generation.get("response_instruction", ""))
    for index, record in enumerate(input_records, start=1):
        meta = record.get("meta", {})
        record_hash = str(meta.get("content_hash") or meta.get("id") or f"ood-{index:03d}")
        fingerprint = _generation_cache_fingerprint(
            record,
            variant=variant,
            backend=backend_config,
            generation=generation,
        )
        cache_path = cache_dir / f"{record_hash}.json"
        candidate = None
        if cache_path.is_file():
            try:
                candidate = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        try:
            messages = prompt_messages(record, response_instruction)
        except Exception:
            # The main loop will turn the precise per-record failure into a
            # benchmark error without preventing other records from batching.
            continue
        prompt_length = sum(len(message.get("content", "")) for message in messages)
        planned.append((
            record_hash,
            cache_path,
            fingerprint,
            messages,
            prompt_length,
            candidate if isinstance(candidate, dict) else None,
        ))

    batch_order = generation.get("batch_order", "manifest")
    if batch_order == "prompt_length_ascending":
        # Scheduling uses prompt text only. Ground-truth answer length is
        # deliberately excluded so batching cannot leak reference properties.
        planned.sort(key=lambda item: (item[4], item[0]))
    elif batch_order != "manifest":
        raise PipelineError(f"Unsupported benchmark batch_order: {batch_order}")

    pending_groups: list[
        tuple[list[tuple[str, Path, str, list[dict[str, str]], int, dict[str, Any] | None]], str]
    ] = []
    for offset in range(0, len(planned), batch_size):
        chunk = planned[offset:offset + batch_size]
        group_hash = content_hash(stable_json({
            "batch_order": batch_order,
            "content_hashes": [item[0] for item in chunk],
        }))
        complete = all(
            isinstance(item[5], dict)
            and item[5].get("fingerprint") == item[2]
            and isinstance(item[5].get("generation"), dict)
            and item[5]["generation"].get("batch_group_hash") == group_hash
            and item[5]["generation"].get("batch_group_position") == position
            and item[5]["generation"].get("generation_batch_size") == len(chunk)
            for position, item in enumerate(chunk)
        )
        if not complete:
            pending_groups.append((chunk, group_hash))

    generated_hashes: set[str] = set()
    total = sum(len(chunk) for chunk, _ in pending_groups)
    completed = 0
    for chunk, group_hash in pending_groups:
        try:
            primary = _backend_generate_batch(
                backend,
                [item[3] for item in chunk],
                generation,
            )
        except Exception as exc:
            LOGGER.warning(
                "Benchmark %s batch %d-%d failed; falling back to sequential generation: %s",
                variant,
                completed + 1,
                completed + len(chunk),
                exc,
            )
            if "out of memory" in str(exc).casefold():
                LOGGER.warning(
                    "Benchmark %s is disabling batching after CUDA OOM; remaining misses will use sequential generation",
                    variant,
                )
                break
            continue
        for position, (
            (record_hash, cache_path, fingerprint, messages, _, _), generated
        ) in enumerate(zip(chunk, primary)):
            generation_result = _generate_with_truncation_retry(
                backend,
                messages,
                generation,
                primary=_generation_payload(generated),
            )
            generation_result["batch_group_hash"] = group_hash
            generation_result["batch_group_position"] = position
            atomic_write_json(
                cache_path,
                {"fingerprint": fingerprint, "generation": generation_result},
            )
            generated_hashes.add(record_hash)
        completed += len(chunk)
        if completed == len(chunk) or completed % 20 == 0 or completed == total:
            LOGGER.info(
                "Benchmark %s deterministic batch cache: %d/%d regenerated records",
                variant,
                completed,
                total,
            )
    return generated_hashes


def _generate_with_truncation_retry(
    backend: Any,
    messages: list[dict[str, str]],
    generation: dict[str, Any],
    *,
    primary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_result = (
        _normalized_generation(primary)
        if primary is not None
        else _generation_payload(backend.generate(messages, generation))
    )
    completion_tokens = primary_result.get("completion_tokens")
    policy = generation.get("retry_on_truncation", {})
    should_retry = (
        isinstance(completion_tokens, int)
        and completion_tokens >= int(generation["max_new_tokens"])
        and primary_result.get("finished_eos") is not True
        and isinstance(policy, dict)
        and policy.get("enabled") is True
    )
    if not should_retry:
        return primary_result

    retry_generation = dict(generation)
    retry_generation["repetition_penalty"] = float(policy["repetition_penalty"])
    retry = backend.generate(messages, retry_generation)
    primary_latency = float(primary_result.get("latency_seconds") or 0.0)
    total_latency = primary_latency + float(retry.latency_seconds)
    primary_tokens = int(primary_result.get("completion_tokens") or 0)
    retry_tokens = int(retry.completion_tokens or 0)
    total_tokens = primary_tokens + retry_tokens
    return {
        "prediction": retry.text,
        "latency_seconds": total_latency,
        "prompt_tokens": retry.prompt_tokens,
        "completion_tokens": retry.completion_tokens,
        "tokens_per_second": total_tokens / total_latency if total_tokens and total_latency > 0 else None,
        "generation_attempts": int(primary_result.get("generation_attempts") or 1) + 1,
        "retry_count": int(primary_result.get("retry_count") or 0) + 1,
        "generated_completion_tokens_total": total_tokens,
        "generation_batch_size": int(primary_result.get("generation_batch_size") or 1),
        "retry_generation_batch_size": 1,
        "finished_eos": getattr(retry, "finished_eos", None),
        "retry_repetition_penalty": float(policy["repetition_penalty"]),
    }


def run_benchmark(
    *,
    variant: str,
    config_path: Path = Path("config/benchmark.yaml"),
    root: Path = Path("."),
    limit: int | None = None,
    skip_judge: bool = False,
    skip_syntax: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    if variant not in config["variants"]:
        raise PipelineError(f"Unknown benchmark variant: {variant}")
    input_records, split_statistics = _load_benchmark_records(config, root, limit)
    backend = build_backend(config["variants"][variant], root)
    cache_dir = root / str(config["outputs"]["work_dir"]) / variant / "generation_cache"
    prefilled_hashes = _prefill_generation_cache(
        backend=backend,
        input_records=input_records,
        variant=variant,
        backend_config=config["variants"][variant],
        generation=config["generation"],
        cache_dir=cache_dir,
    )
    results: list[dict[str, Any]] = []
    generated_count = 0
    cached_count = 0
    compatible_cached_count = 0
    retry_policy_reused_count = 0
    error_count = 0
    for index, record in enumerate(input_records, start=1):
        meta = record.get("meta", {})
        content_hash = str(meta.get("content_hash") or meta.get("id") or f"ood-{index:03d}")
        category = str(meta.get("category", "out_of_domain"))
        reference = role_content(record, "assistant")
        question = role_content(record, "user")
        output: dict[str, Any] = {
            "content_hash": content_hash,
            "category": category,
            "source": str(meta.get("source", "out_of_domain_fixture")),
            "question": question,
            "reference": reference,
            "prediction": "",
            "latency_seconds": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "tokens_per_second": None,
            "truncated": None,
            "finished_eos": None,
            "exact_match": None,
            "semantic_similarity": None,
            "syntax_valid": None,
            "syntax_reason": None,
            "judge": None,
            "error": None,
        }
        fingerprint = _generation_cache_fingerprint(
            record,
            variant=variant,
            backend=config["variants"][variant],
            generation=config["generation"],
        )
        cache_path = cache_dir / f"{content_hash}.json"
        try:
            cached_generation = None
            compatible_prior_limit = None
            legacy_retry_primary = None
            legacy_policy_reused = False
            if cache_path.is_file():
                try:
                    candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if isinstance(candidate, dict):
                    cached_generation, compatible_prior_limit = _compatible_cached_generation(
                        candidate,
                        record,
                        variant=variant,
                        backend=config["variants"][variant],
                        generation=config["generation"],
                        current_fingerprint=fingerprint,
                    )
                    if cached_generation is None:
                        legacy_generation, legacy_action = _legacy_retry_policy_cache(
                            candidate,
                            record,
                            variant=variant,
                            backend=config["variants"][variant],
                            generation=config["generation"],
                        )
                        if legacy_action == "reuse":
                            cached_generation = legacy_generation
                            retry_policy_reused_count += 1
                            legacy_policy_reused = True
                        elif legacy_action == "retry":
                            legacy_retry_primary = legacy_generation
            if isinstance(cached_generation, dict):
                cached_generation = _normalized_generation(cached_generation)
                output.update(cached_generation)
                if content_hash in prefilled_hashes:
                    generated_count += 1
                else:
                    cached_count += 1
                if compatible_prior_limit is not None or legacy_policy_reused:
                    compatible_cached_count += 1
                    atomic_write_json(
                        cache_path,
                        {
                            "fingerprint": fingerprint,
                            "generation": cached_generation,
                            "reused_from_max_new_tokens": compatible_prior_limit,
                            "reused_before_retry_policy": compatible_prior_limit is None,
                        },
                    )
            else:
                generation_result = _generate_with_truncation_retry(
                    backend,
                    prompt_messages(record, str(config["generation"].get("response_instruction", ""))),
                    config["generation"],
                    primary=legacy_retry_primary,
                )
                output.update(generation_result)
                atomic_write_json(
                    cache_path,
                    {"fingerprint": fingerprint, "generation": generation_result},
                )
                generated_count += 1
            completion_tokens = output.get("completion_tokens")
            output["truncated"] = (
                completion_tokens is not None
                and int(completion_tokens) >= int(config["generation"]["max_new_tokens"])
                and output.get("finished_eos") is not True
            )
            if category in {"comando_cli", "out_of_domain"}:
                output["exact_match"] = exact_match(reference, str(output["prediction"]))
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
            error_count += 1
        results.append(output)
        if index == 1 or index % 10 == 0 or index == len(input_records):
            LOGGER.info(
                "Benchmark %s progress: %d/%d (generated=%d cached=%d errors=%d)",
                variant,
                index,
                len(input_records),
                generated_count,
                cached_count,
                error_count,
            )

    scored = [record for record in results if not record["error"]]
    similarities = semantic_scores(
        [str(record["reference"]) for record in scored],
        [str(record["prediction"]) for record in scored],
        config["semantic_similarity"],
    )
    for record, score in zip(scored, similarities):
        record["semantic_similarity"] = float(score)

    if config["syntax_validation"]["enabled"] and not skip_syntax:
        syntax = syntax_scores(results, variant=variant, benchmark_config=config, root=root)
        for record in results:
            if record["content_hash"] in syntax:
                record["syntax_valid"] = syntax[record["content_hash"]]["valid"]
                record["syntax_reason"] = syntax[record["content_hash"]]["reason"]

    if config["llm_judge"]["enabled"] and not skip_judge:
        judge_hashes = _validated_judge_hashes(
            config, root, input_records, split_statistics
        )
        prompt = (root / str(config["llm_judge"]["prompt"])).read_text(encoding="utf-8")
        judge = JudgeClient(config["llm_judge"], prompt)
        for record in results:
            if record["error"]:
                continue
            if record["category"] != "out_of_domain" and record["content_hash"] not in judge_hashes:
                continue
            record["judge"] = judge.score(
                question=str(record["question"]),
                reference=str(record["reference"]),
                candidate=str(record["prediction"]),
            )

    generated_records = [record for record in results if not record["error"]]
    truncated_count = sum(bool(record.get("truncated")) for record in generated_records)
    truncation_rate = truncated_count / len(generated_records) if generated_records else 1.0
    truncation_threshold = float(config["generation"]["max_truncation_rate"])
    truncation_audit = {
        "max_allowed_rate": truncation_threshold,
        "truncated": truncated_count,
        "generated": len(generated_records),
        "rate": truncation_rate,
        "passed": truncation_rate <= truncation_threshold,
    }
    completion = {
        "full_test": limit is None,
        "generation": (
            not any(record["error"] for record in results)
            and truncation_audit["passed"]
        ),
        "syntax": not config["syntax_validation"]["enabled"] or not skip_syntax,
        "judge": not config["llm_judge"]["enabled"] or not skip_judge,
    }
    payload = {
        "version": int(config["version"]),
        "created_at": utc_now_iso(),
        "variant": variant,
        "backend": config["variants"][variant],
        "model_provenance": getattr(backend, "provenance", None),
        "runtime": getattr(backend, "runtime", None),
        "generation_cache": {
            "path": cache_dir.relative_to(root).as_posix(),
            "generated": generated_count,
            "cached": cached_count,
            "compatible_reused": compatible_cached_count,
            "retry_policy_reused": retry_policy_reused_count,
        },
        "truncation_audit": truncation_audit,
        "provisional": _is_provisional(split_statistics, completion),
        "completion": completion,
        "split_test_sha256": split_statistics["outputs"]["test"]["sha256"],
        "generation": config["generation"],
        "semantic_similarity_config": config["semantic_similarity"],
        "syntax_provenance": syntax_provenance(config, root),
        "judge_provenance": judge_provenance(config, root),
        "metrics": aggregate_results(results),
        "records": results,
    }
    atomic_write_json(_output_path(config, variant, root), payload)
    if variant.startswith("finetuned_"):
        _update_combined(config, root)
    return payload


def run_benchmark_judge(
    *, variant: str, config_path: Path = Path("config/benchmark.yaml"), root: Path = Path("."),
) -> dict[str, Any]:
    """Judge saved predictions after the evaluated model has released the GPU."""
    root = root.resolve()
    config = load_yaml(root / config_path)
    if variant not in config["variants"]:
        raise PipelineError(f"Unknown benchmark variant: {variant}")
    output_path = _output_path(config, variant, root)
    if not output_path.is_file():
        raise PipelineError(f"Benchmark predictions are missing: {output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("variant") != variant:
        raise PipelineError(f"Benchmark result variant mismatch in {output_path}")
    if payload.get("generation") != config["generation"]:
        raise PipelineError("Saved predictions use different generation parameters")
    if payload.get("semantic_similarity_config") != config.get("semantic_similarity"):
        raise PipelineError("Saved predictions use a different semantic similarity model")

    expected_records, split_statistics = _load_benchmark_records(config, root, limit=None)
    expected_hashes = [
        str(record.get("meta", {}).get("content_hash")) for record in expected_records
    ]
    result_hashes = [str(record.get("content_hash")) for record in payload.get("records", [])]
    if result_hashes != expected_hashes:
        raise PipelineError("Saved predictions do not exactly match the frozen full benchmark order")
    if payload.get("split_test_sha256") != split_statistics["outputs"]["test"]["sha256"]:
        raise PipelineError("Saved predictions use a different frozen test split")
    expected_judge_provenance = judge_provenance(config, root)
    if payload.get("judge_provenance") != expected_judge_provenance:
        raise PipelineError("Saved predictions use a different judge prompt or manifest")
    completion = dict(payload.get("completion") or {})
    if not completion.get("full_test", False):
        raise PipelineError("Cannot finalize judging for a limited benchmark run")
    if not completion.get("generation", False):
        raise PipelineError("Cannot finalize judging before successful generation")
    if completion.get("syntax") and payload.get("syntax_provenance") != syntax_provenance(config, root):
        raise PipelineError("Saved predictions use a different syntax validator configuration")
    if any(record.get("error") for record in payload.get("records", [])):
        raise PipelineError("Successful generation marker is inconsistent with record errors")

    judge_hashes = _validated_judge_hashes(
        config, root, expected_records, split_statistics
    )
    prompt = (root / str(config["llm_judge"]["prompt"])).read_text(encoding="utf-8")
    judge = JudgeClient(config["llm_judge"], prompt)
    current_judge_runtime = getattr(judge, "provenance", None)
    previous_judge_runtime = payload.get("judge_runtime")
    if previous_judge_runtime != current_judge_runtime and any(
        isinstance(record.get("judge"), dict) for record in payload["records"]
    ):
        LOGGER.warning("Judge runtime changed; invalidating cached judge scores")
        for record in payload["records"]:
            record["judge"] = None
    payload["judge_runtime"] = current_judge_runtime
    judged = 0
    cached = 0
    expected_judged = sum(
        record.get("category") == "out_of_domain"
        or str(record.get("content_hash")) in judge_hashes
        for record in payload["records"]
    )
    for record in payload["records"]:
        should_judge = (
            not record.get("error")
            and (
                record.get("category") == "out_of_domain"
                or str(record.get("content_hash")) in judge_hashes
            )
        )
        if not should_judge:
            continue
        if isinstance(record.get("judge"), dict):
            try:
                record["judge"] = parse_judge_payload(record["judge"])
            except PipelineError:
                record["judge"] = None
            else:
                cached += 1
                continue
        record["judge"] = judge.score(
            question=str(record["question"]),
            reference=str(record["reference"]),
            candidate=str(record["prediction"]),
        )
        judged += 1
        payload["metrics"] = aggregate_results(payload["records"])
        atomic_write_json(output_path, payload)

    valid_judged = 0
    for record in payload["records"]:
        if not (
            record.get("category") == "out_of_domain"
            or str(record.get("content_hash")) in judge_hashes
        ):
            continue
        try:
            record["judge"] = parse_judge_payload(record.get("judge") or {})
        except PipelineError as exc:
            raise PipelineError("Judge completion contains an invalid or missing score") from exc
        valid_judged += 1
    if valid_judged != expected_judged:
        raise PipelineError(
            f"Judge completion count mismatch: {valid_judged} != {expected_judged}"
        )
    completion["judge"] = True
    payload["completion"] = completion
    payload["syntax_provenance"] = syntax_provenance(config, root)
    payload["provisional"] = _is_provisional(split_statistics, completion)
    payload["judge_completed_at"] = utc_now_iso()
    payload["metrics"] = aggregate_results(payload["records"])
    atomic_write_json(output_path, payload)
    if variant.startswith("finetuned_"):
        _update_combined(config, root)
    return {
        "variant": variant,
        "judged": judged,
        "cached": cached,
        "provisional": payload["provisional"],
        "metrics": payload["metrics"],
    }


def run_benchmark_syntax(
    *, variant: str, config_path: Path = Path("config/benchmark.yaml"), root: Path = Path("."),
) -> dict[str, Any]:
    """Validate saved predictions on the Docker host after GPU generation."""
    root = root.resolve()
    config = load_yaml(root / config_path)
    if variant not in config["variants"]:
        raise PipelineError(f"Unknown benchmark variant: {variant}")
    output_path = _output_path(config, variant, root)
    if not output_path.is_file():
        raise PipelineError(f"Benchmark predictions are missing: {output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("variant") != variant:
        raise PipelineError(f"Benchmark result variant mismatch in {output_path}")
    if payload.get("generation") != config["generation"]:
        raise PipelineError("Saved predictions use different generation parameters")
    if payload.get("semantic_similarity_config") != config.get("semantic_similarity"):
        raise PipelineError("Saved predictions use a different semantic similarity model")

    expected_records, split_statistics = _load_benchmark_records(config, root, limit=None)
    expected_hashes = [
        str(record.get("meta", {}).get("content_hash")) for record in expected_records
    ]
    result_hashes = [str(record.get("content_hash")) for record in payload.get("records", [])]
    if result_hashes != expected_hashes:
        raise PipelineError("Saved predictions do not exactly match the frozen full benchmark order")
    if payload.get("split_test_sha256") != split_statistics["outputs"]["test"]["sha256"]:
        raise PipelineError("Saved predictions use a different frozen test split")
    completion = dict(payload.get("completion") or {})
    if not completion.get("full_test", False):
        raise PipelineError("Cannot finalize syntax for a limited benchmark run")
    if not completion.get("generation", False):
        raise PipelineError("Cannot finalize syntax before successful generation")
    if any(record.get("error") for record in payload.get("records", [])):
        raise PipelineError("Successful generation marker is inconsistent with record errors")

    syntax = syntax_scores(payload["records"], variant=variant, benchmark_config=config, root=root)
    expected_syntax_hashes = {
        str(record["content_hash"])
        for record in payload["records"]
        if record.get("category") in {"generacion_yaml", "dockerfile"}
    }
    if set(syntax) != expected_syntax_hashes:
        raise PipelineError("Syntax results do not exactly cover every YAML and Dockerfile record")
    try:
        invalid_syntax_decision = any(
            float(value.get("valid", -1.0)) not in {0.0, 1.0}
            for value in syntax.values()
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise PipelineError("Syntax results must contain binary validity decisions") from exc
    if invalid_syntax_decision:
        raise PipelineError("Syntax results must contain binary validity decisions")
    for record in payload["records"]:
        content_hash = str(record.get("content_hash"))
        if content_hash in syntax:
            record["syntax_valid"] = syntax[content_hash]["valid"]
            record["syntax_reason"] = syntax[content_hash]["reason"]
    completion["syntax"] = True
    payload["completion"] = completion
    payload["syntax_provenance"] = syntax_provenance(config, root)
    payload["provisional"] = _is_provisional(split_statistics, completion)
    payload["syntax_completed_at"] = utc_now_iso()
    payload["metrics"] = aggregate_results(payload["records"])
    atomic_write_json(output_path, payload)
    if variant.startswith("finetuned_"):
        _update_combined(config, root)
    return {
        "variant": variant,
        "checked": len(syntax),
        "valid": sum(value["valid"] == 1.0 for value in syntax.values()),
        "provisional": payload["provisional"],
        "metrics": payload["metrics"],
    }
