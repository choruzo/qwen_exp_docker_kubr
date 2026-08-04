from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from ..config import load_yaml, validate_sources_config
from ..errors import ConfigError
from .base import BaseExtractor
from .git_markdown import FailureStoryIndexExtractor, GitMarkdownExtractor


LOGGER = logging.getLogger(__name__)


def extractor_class(kind: str) -> type[BaseExtractor]:
    if kind == "git_markdown":
        return GitMarkdownExtractor
    if kind == "failure_story_index":
        return FailureStoryIndexExtractor
    if kind == "huggingface_dataset":
        from .huggingface import HuggingFaceExtractor

        return HuggingFaceExtractor
    if kind == "github_issues":
        from .github_issues import GitHubIssuesExtractor

        return GitHubIssuesExtractor
    if kind == "stackoverflow_bigquery":
        from .stackoverflow import StackOverflowBigQueryExtractor

        return StackOverflowBigQueryExtractor
    raise ConfigError(f"Unsupported extractor kind: {kind}")


def _selected_names(
    all_sources: dict[str, Any], selected_sources: Iterable[str] | None, include_quarantine: bool
) -> list[str]:
    if selected_sources:
        selected = list(dict.fromkeys(selected_sources))
        unknown = sorted(set(selected) - all_sources.keys())
        if unknown:
            raise ConfigError(f"Unknown source ids: {unknown}")
    else:
        selected = list(all_sources)
    return [
        name
        for name in selected
        if all_sources[name].get("enabled", True)
        and (include_quarantine or all_sources[name].get("tier") != "quarantine")
    ]


def run_extraction(
    *,
    config_path: Path,
    selected_sources: Iterable[str] | None,
    force: bool,
    include_quarantine: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_sources_config(config, require_approved=True)
    sources: dict[str, Any] = config["sources"]
    names = _selected_names(sources, selected_sources, include_quarantine)
    if dry_run:
        return {
            "dry_run": True,
            "selected": names,
            "excluded_quarantine": [
                name for name, source in sources.items() if source.get("tier") == "quarantine" and name not in names
            ],
        }

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name in names:
        source = sources[name]
        LOGGER.info("Starting extraction: %s", name)
        try:
            extractor = extractor_class(str(source["kind"]))(
                source_name=name,
                source_config=source,
                root_config=config,
                force=force,
            )
            results.append(extractor.extract().to_dict())
        except Exception as exc:  # batch mode preserves successful source outputs
            LOGGER.exception("Extraction failed for %s", name)
            failures.append({"source": name, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "dry_run": False,
        "selected": names,
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
