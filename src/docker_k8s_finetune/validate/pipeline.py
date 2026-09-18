from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, file_sha256, read_jsonl
from .core import assistant_content, extract_dockerfile, extract_yaml, parse_json_output


_HADOLINT_SEVERITY = {"ignore": 0, "none": 0, "style": 1, "info": 2, "warning": 3, "error": 4}


def _hadolint_issue_fails(issue: dict[str, Any], threshold: str) -> bool:
    configured = str(threshold).lower()
    if configured not in _HADOLINT_SEVERITY:
        raise PipelineError(f"Unsupported hadolint failure threshold: {threshold}")
    level = str(issue.get("level", "error")).lower()
    return _HADOLINT_SEVERITY.get(level, _HADOLINT_SEVERITY["error"]) >= _HADOLINT_SEVERITY[configured]


def _safe_recreate(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise PipelineError(f"Refusing to recreate unsafe validation directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise PipelineError(f"Required executable is not available: {command[0]}") from exc


def _docker_mount(path: Path) -> str:
    return f"type=bind,source={path.resolve()},target=/work,readonly"


def _validate_kubernetes(work: Path, files: list[Path], config: dict[str, Any]) -> dict[str, str]:
    if not files:
        return {}
    tool = config["kubeconform"]
    command = [
        "docker", "run", "--rm", "--mount", _docker_mount(work), str(tool["image"]),
        "-kubernetes-version", str(tool["kubernetes_version"]), "-output", "json", "-summary",
    ]
    if tool.get("strict", False):
        command.append("-strict")
    if tool.get("ignore_missing_schemas", False):
        command.append("-ignore-missing-schemas")
    command.append("/work/yaml")
    result = _run(command)
    parsed = parse_json_output(result.stdout)
    if not parsed:
        raise PipelineError(f"kubeconform produced no JSON output: {result.stderr.strip()}")
    payload = parsed[0]
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise PipelineError("kubeconform JSON output has no summary")
    invalid: dict[str, str] = {}
    for item in payload.get("resources", []):
        if not isinstance(item, dict):
            continue
        filename = Path(str(item.get("filename", ""))).name
        status = str(item.get("status", ""))
        if status != "statusValid":
            invalid[filename] = str(item.get("msg") or status or "kubeconform_failed")
    if result.returncode not in (0, 1):
        raise PipelineError(f"kubeconform failed with exit {result.returncode}: {result.stderr.strip()}")
    return invalid


def _validate_dockerfiles(work: Path, files: list[Path], config: dict[str, Any]) -> dict[str, str]:
    if not files:
        return {}
    tool = config["hadolint"]
    batch_size = int(tool.get("batch_size", 200))
    invalid: dict[str, list[str]] = {}
    for offset in range(0, len(files), batch_size):
        container_files = [
            f"/work/dockerfile/{path.name}" for path in files[offset : offset + batch_size]
        ]
        command = [
            "docker", "run", "--rm", "--mount", _docker_mount(work), str(tool["image"]),
            "/bin/hadolint", "--format", "json", "--failure-threshold", str(tool["failure_threshold"]),
            *container_files,
        ]
        result = _run(command)
        issues = parse_json_output(result.stdout)
        batch_invalid = 0
        for issue in issues:
            if not _hadolint_issue_fails(issue, str(tool["failure_threshold"])):
                continue
            filename = Path(str(issue.get("file", ""))).name
            message = (
                f"{issue.get('code', 'hadolint')}[{issue.get('level', 'error')}]: "
                f"{issue.get('message', 'lint failure')}"
            )
            invalid.setdefault(filename, []).append(message)
            batch_invalid += 1
        if result.returncode not in (0, 1):
            raise PipelineError(f"hadolint failed with exit {result.returncode}: {result.stderr.strip()}")
        if result.returncode == 1 and not batch_invalid:
            raise PipelineError(f"hadolint failed without structured issues: {result.stderr.strip()}")
    return {filename: "; ".join(messages) for filename, messages in invalid.items()}


def run_syntax_validation(
    *,
    config_path: Path = Path("config/validation.yaml"),
    input_override: Path | None = None,
    root: Path = Path("."),
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    input_path = root / (input_override or Path(config["input"]))
    if not input_path.exists():
        raise PipelineError(f"Validation input does not exist: {input_path}")
    work = root / config["outputs"]["work_dir"]
    _safe_recreate(work, root)
    yaml_dir = work / "yaml"
    dockerfile_dir = work / "dockerfile"
    yaml_dir.mkdir()
    dockerfile_dir.mkdir()

    records = list(read_jsonl(input_path))
    candidates: dict[str, tuple[dict[str, Any], str, str]] = {}
    missing: dict[str, str] = {}
    category_changes: Counter[str] = Counter()
    yaml_files: list[Path] = []
    dockerfile_files: list[Path] = []
    reclassification = config.get("category_reclassification", {})
    dockerfile_from_categories = {
        str(value) for value in reclassification.get("dockerfile_from_categories", [])
    }
    missing_dockerfile_fallback = str(
        reclassification.get("missing_dockerfile_fallback", "")
    )
    for index, record in enumerate(records):
        meta = record.get("meta", {})
        category = str(meta.get("category", ""))
        content_text = assistant_content(record)
        dockerfile_content = extract_dockerfile(content_text)
        if dockerfile_content is not None and category in dockerfile_from_categories:
            original = category
            category = "dockerfile"
            meta["category"] = category
            meta["category_reclassification"] = {
                "from": original,
                "to": category,
                "reason": "dockerfile_block_detected",
            }
            category_changes[f"{original}->dockerfile"] += 1
        elif category == "dockerfile" and dockerfile_content is None and missing_dockerfile_fallback:
            category = missing_dockerfile_fallback
            meta["category"] = category
            meta["category_reclassification"] = {
                "from": "dockerfile",
                "to": category,
                "reason": "dockerfile_block_missing",
            }
            category_changes[f"dockerfile->{category}"] += 1
        identifier = str(meta.get("content_hash") or meta.get("source_record_id") or f"record-{index}")
        filename_base = f"{index:06d}-{identifier[:16]}"
        if category == "generacion_yaml":
            content = extract_yaml(content_text)
            if content is None:
                missing[identifier] = "yaml_block_missing"
                continue
            path = yaml_dir / f"{filename_base}.yaml"
            path.write_text(content + "\n", encoding="utf-8", newline="\n")
            yaml_files.append(path)
            candidates[path.name] = (record, identifier, "kubeconform")
        elif category == "dockerfile":
            content = dockerfile_content
            if content is None:
                missing[identifier] = "dockerfile_block_missing"
                continue
            path = dockerfile_dir / f"{filename_base}.Dockerfile"
            path.write_text(content + "\n", encoding="utf-8", newline="\n")
            dockerfile_files.append(path)
            candidates[path.name] = (record, identifier, "hadolint")

    failures = _validate_kubernetes(work, yaml_files, config)
    failures.update(_validate_dockerfiles(work, dockerfile_files, config))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_source_input: Counter[str] = Counter()
    by_source_rejected: Counter[str] = Counter()
    candidate_by_id = {identifier: filename for filename, (_, identifier, _) in candidates.items()}
    candidate_details = {identifier: tool for _, identifier, tool in candidates.values()}
    for record in records:
        meta = record.setdefault("meta", {})
        source = str(meta.get("source", "unknown"))
        by_source_input[source] += 1
        identifier = str(meta.get("content_hash") or meta.get("source_record_id"))
        reason = missing.get(identifier)
        filename = candidate_by_id.get(identifier)
        if filename and filename in failures:
            reason = failures[filename]
        if reason:
            by_source_rejected[source] += 1
            meta["syntax_validation"] = {"status": "rejected", "reason": reason}
            rejected.append(record)
            continue
        tool = candidate_details.get(identifier)
        meta["syntax_validation"] = {
            "status": "valid" if tool else "not_applicable",
            "tool": tool,
        }
        accepted.append(record)

    outputs = config["outputs"]
    accepted_count, accepted_hash = atomic_write_jsonl(root / outputs["accepted"], accepted)
    rejected_count, rejected_hash = atomic_write_jsonl(root / outputs["rejected"], rejected)
    report = {
        "version": int(config["version"]),
        "input": input_path.relative_to(root).as_posix(),
        "input_sha256": file_sha256(input_path),
        "provisional_input_override": input_override is not None,
        "tools": {
            "kubeconform": config["kubeconform"],
            "hadolint": config["hadolint"],
        },
        "counts": {
            "input": len(records), "accepted": accepted_count, "rejected": rejected_count,
            "yaml_candidates": len(yaml_files), "dockerfile_candidates": len(dockerfile_files),
        },
        "input_by_source": dict(sorted(by_source_input.items())),
        "rejected_by_source": dict(sorted(by_source_rejected.items())),
        "category_reclassification": dict(sorted(category_changes.items())),
        "outputs": {
            "accepted": {"path": outputs["accepted"], "sha256": accepted_hash},
            "rejected": {"path": outputs["rejected"], "sha256": rejected_hash},
        },
    }
    atomic_write_json(root / outputs["report"], report)
    return report
