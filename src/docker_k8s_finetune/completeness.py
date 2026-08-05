from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config import load_yaml
from .io import content_hash, file_sha256, stable_json


def _issue(issues: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    issues.append({"code": code, "message": message, **details})


def _load_report(path: Path, issues: list[dict[str, Any]], code: str) -> dict[str, Any] | None:
    if not path.is_file():
        _issue(issues, code, f"Required report does not exist: {path}", path=str(path))
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _issue(issues, code, f"Cannot read report {path}: {exc}", path=str(path))
        return None
    if not isinstance(value, dict):
        _issue(issues, code, f"Report is not a JSON object: {path}", path=str(path))
        return None
    return value


def _verify_artifact(
    *, root: Path, relative: str, expected_hash: str | None, issues: list[dict[str, Any]],
    missing_code: str, hash_code: str, verify_hashes: bool,
) -> str | None:
    path = root / relative
    if not path.is_file():
        _issue(issues, missing_code, f"Required artifact does not exist: {relative}", path=relative)
        return None
    if not verify_hashes:
        return None
    actual = file_sha256(path)
    if expected_hash and actual != expected_hash:
        _issue(
            issues, hash_code, f"Artifact hash differs from its recorded lineage: {relative}",
            path=relative, expected_sha256=expected_hash, actual_sha256=actual,
        )
    return actual


def assess_pipeline_completeness(
    *, root: Path = Path("."), sources_path: Path = Path("config/sources.yaml"),
    normalization_path: Path = Path("config/normalization.yaml"),
    dedupe_path: Path = Path("config/dedupe.yaml"),
    validation_path: Path = Path("config/validation.yaml"),
    cleaning_report_path: Path = Path("data/interim/reports/cleaning_report.json"),
    verify_hashes: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    sources = load_yaml(root / sources_path)
    normalization = load_yaml(root / normalization_path)
    dedupe = load_yaml(root / dedupe_path)
    validation = load_yaml(root / validation_path)
    issues: list[dict[str, Any]] = []
    required_environment: set[str] = set()

    expected_sources: list[str] = []
    extracted_sources: list[str] = []
    cleaning_report = _load_report(root / cleaning_report_path, issues, "cleaning_report_missing")
    cleaning_inputs = {
        str(item.get("path")): str(item.get("sha256"))
        for item in (cleaning_report or {}).get("inputs", [])
        if isinstance(item, Mapping)
    }
    defaults = sources.get("defaults", {})
    output_template = str(defaults.get("output_template", "data/interim/extracted/{source}.jsonl"))
    manifest_suffix = str(defaults.get("idempotency", {}).get("manifest_suffix", ".manifest.json"))
    for name, source in sources.get("sources", {}).items():
        if not source.get("enabled", True):
            continue
        if source.get("tier") == "quarantine" or source.get("include_in_final") is False:
            continue
        expected_sources.append(str(name))
        relative = str(source.get("destination") or output_template.format(source=name))
        manifest_relative = relative + manifest_suffix
        output = root / relative
        manifest_path = root / manifest_relative
        if not output.is_file():
            _issue(issues, "source_output_missing", f"Extraction output is missing for {name}", source=name, path=relative)
            for key in ("token_env", "project_env"):
                if source.get(key):
                    required_environment.add(str(source[key]))
            continue
        manifest = _load_report(manifest_path, issues, "source_manifest_missing")
        if manifest is None:
            continue
        extracted_sources.append(str(name))
        if manifest.get("source") != name or int(manifest.get("records", 0)) <= 0:
            _issue(issues, "source_manifest_invalid", f"Extraction manifest is invalid for {name}", source=name)
        expected_config_hash = content_hash(stable_json(source))
        if manifest.get("source_config_sha256") != expected_config_hash:
            _issue(issues, "source_config_drift", f"Source config changed after extracting {name}", source=name)
        actual_hash = _verify_artifact(
            root=root, relative=relative, expected_hash=str(manifest.get("sha256") or ""), issues=issues,
            missing_code="source_output_missing", hash_code="source_output_hash_mismatch",
            verify_hashes=verify_hashes,
        )
        cleaned_hash = cleaning_inputs.get(relative)
        if cleaning_report is not None and (not cleaned_hash or (actual_hash and cleaned_hash != actual_hash)):
            _issue(
                issues, "cleaning_lineage_stale",
                f"Cleaning report does not cover the current extraction output for {name}",
                source=name, path=relative,
            )

    reverse = normalization["reverse_instruction"]
    review_path = root / str(reverse["review_config"])
    review = load_yaml(review_path) if review_path.is_file() else {}
    if review.get("status") != "approved":
        _issue(issues, "reverse_review_pending", f"Reverse-instruction review is not approved: {review_path}")
    sample_relative = str(reverse["generated_sample"])
    sample_path = root / sample_relative
    if review.get("status") == "approved":
        if not sample_path.is_file():
            _issue(issues, "reverse_sample_missing", f"Approved reverse sample is missing: {sample_relative}")
        elif verify_hashes and review.get("sample_sha256") != file_sha256(sample_path):
            _issue(issues, "reverse_sample_hash_mismatch", "Approved reverse sample hash no longer matches")
    full_relative = str(reverse["generated_full"])
    if not (root / full_relative).is_file():
        _issue(issues, "reverse_full_missing", f"Full reverse-instruction output is missing: {full_relative}")
        required_environment.update((str(reverse["base_url_env"]), str(reverse["model_env"])))

    exact_report_path = root / str(dedupe["exact"]["report"])
    exact_report = _load_report(exact_report_path, issues, "exact_dedupe_report_missing")
    if exact_report is not None:
        if exact_report.get("missing_inputs"):
            _issue(
                issues, "exact_dedupe_incomplete", "Exact dedupe ran without all normalized inputs",
                missing_inputs=list(exact_report["missing_inputs"]),
            )
        lineage = {
            str(item.get("path")): str(item.get("sha256"))
            for item in exact_report.get("input_files", []) if isinstance(item, Mapping)
        }
        if not lineage:
            _issue(issues, "exact_dedupe_lineage_missing", "Exact dedupe report has no input hashes")
        for relative in map(str, dedupe["inputs"]):
            path = root / relative
            if not path.is_file():
                continue
            actual = file_sha256(path) if verify_hashes else None
            if relative not in lineage or (actual and lineage[relative] != actual):
                _issue(issues, "exact_dedupe_lineage_stale", f"Exact dedupe is stale for {relative}", path=relative)
        output = exact_report.get("output", {})
        _verify_artifact(
            root=root, relative=str(output.get("path") or dedupe["exact"]["output"]),
            expected_hash=str(output.get("sha256") or ""), issues=issues,
            missing_code="exact_dedupe_output_missing", hash_code="exact_dedupe_output_hash_mismatch",
            verify_hashes=verify_hashes,
        )

    approximate_report_path = root / str(dedupe["approximate"]["report"])
    approximate_report = _load_report(approximate_report_path, issues, "approximate_dedupe_report_missing")
    if approximate_report is not None:
        input_info = approximate_report.get("input", {})
        exact_relative = str(dedupe["exact"]["output"])
        exact_path = root / exact_relative
        actual_exact = file_sha256(exact_path) if verify_hashes and exact_path.is_file() else None
        if input_info.get("path") != exact_relative or (actual_exact and input_info.get("sha256") != actual_exact):
            _issue(issues, "approximate_dedupe_lineage_stale", "Approximate dedupe does not match exact output")
        for key, missing_code, hash_code in (
            ("output", "approximate_dedupe_output_missing", "approximate_dedupe_output_hash_mismatch"),
            ("embeddings", "semantic_embeddings_missing", "semantic_embeddings_hash_mismatch"),
        ):
            artifact = approximate_report.get(key, {})
            configured = dedupe["approximate"]["output" if key == "output" else "embeddings"]
            _verify_artifact(
                root=root, relative=str(artifact.get("path") or configured),
                expected_hash=str(artifact.get("sha256") or ""), issues=issues,
                missing_code=missing_code, hash_code=hash_code, verify_hashes=verify_hashes,
            )

    validation_report_path = root / str(validation["outputs"]["report"])
    validation_report = _load_report(validation_report_path, issues, "syntax_validation_report_missing")
    if validation_report is not None:
        if validation_report.get("provisional_input_override"):
            _issue(issues, "syntax_validation_provisional", "Syntax validation used a provisional input override")
        expected_input = str(validation["input"])
        input_path = root / expected_input
        actual_input = file_sha256(input_path) if verify_hashes and input_path.is_file() else None
        if validation_report.get("input") != expected_input or (
            actual_input and validation_report.get("input_sha256") != actual_input
        ):
            _issue(issues, "syntax_validation_lineage_stale", "Syntax validation does not match semantic dedupe output")
        accepted = validation_report.get("outputs", {}).get("accepted", {})
        _verify_artifact(
            root=root, relative=str(accepted.get("path") or validation["outputs"]["accepted"]),
            expected_hash=str(accepted.get("sha256") or ""), issues=issues,
            missing_code="validated_output_missing", hash_code="validated_output_hash_mismatch",
            verify_hashes=verify_hashes,
        )

    missing_environment = sorted(name for name in required_environment if not os.getenv(name, "").strip())
    return {
        "version": 1,
        "ready_for_final_split": not issues,
        "expected_final_sources": sorted(expected_sources),
        "extracted_final_sources": sorted(extracted_sources),
        "missing_environment": missing_environment,
        "issues": issues,
    }
