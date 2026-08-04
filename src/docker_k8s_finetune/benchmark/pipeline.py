from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, read_jsonl
from ..schema import utc_now_iso
from .backends import build_backend
from .core import aggregate_results, exact_match, prompt_messages, role_content
from .scoring import JudgeClient, semantic_scores, syntax_scores


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineError(f"Frozen benchmark manifest does not exist: {path}")
    values = list(read_jsonl(path))
    if not values:
        raise PipelineError(f"Frozen benchmark manifest is empty: {path}")
    return values


def _load_benchmark_records(config: dict[str, Any], root: Path, limit: int | None) -> list[dict[str, Any]]:
    test_path = root / str(config["inputs"]["test"])
    if not test_path.is_file():
        raise PipelineError(f"Held-out test set does not exist: {test_path}")
    by_hash = {str(record["meta"]["content_hash"]): record for record in read_jsonl(test_path)}
    manifest = _load_manifest(root / str(config["inputs"]["test_manifest"]))
    manifest_hashes = [str(item["content_hash"]) for item in manifest]
    if set(manifest_hashes) != set(by_hash):
        raise PipelineError("Frozen test manifest does not exactly match data/processed/test.jsonl")
    selected = [by_hash[value] for value in manifest_hashes]
    if limit is not None:
        selected = selected[:limit]
    ood_path = root / str(config["inputs"]["out_of_domain"])
    ood = list(read_jsonl(ood_path)) if ood_path.is_file() else []
    if not ood:
        raise PipelineError(f"Out-of-domain fixture is missing or empty: {ood_path}")
    return selected + ood


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
    input_records = _load_benchmark_records(config, root, limit)
    backend = build_backend(config["variants"][variant], root)
    results: list[dict[str, Any]] = []
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
            "exact_match": None,
            "semantic_similarity": None,
            "syntax_valid": None,
            "syntax_reason": None,
            "judge": None,
            "error": None,
        }
        try:
            generated = backend.generate(prompt_messages(record), config["generation"])
            output.update({
                "prediction": generated.text,
                "latency_seconds": generated.latency_seconds,
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "tokens_per_second": generated.tokens_per_second,
            })
            if category in {"comando_cli", "out_of_domain"}:
                output["exact_match"] = exact_match(reference, generated.text)
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        results.append(output)

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
        judge_hashes = {
            str(item["content_hash"])
            for item in _load_manifest(root / str(config["inputs"]["judge_manifest"]))
        }
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

    payload = {
        "version": int(config["version"]),
        "created_at": utc_now_iso(),
        "variant": variant,
        "backend": config["variants"][variant],
        "provisional": bool(limit is not None or skip_judge or skip_syntax),
        "generation": config["generation"],
        "metrics": aggregate_results(results),
        "records": results,
    }
    atomic_write_json(_output_path(config, variant, root), payload)
    if variant.startswith("finetuned_"):
        _update_combined(config, root)
    return payload
