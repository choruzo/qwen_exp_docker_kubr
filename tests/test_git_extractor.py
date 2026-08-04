from __future__ import annotations

import subprocess
from pathlib import Path

from docker_k8s_finetune.extract.git_markdown import GitMarkdownExtractor, split_markdown
from docker_k8s_finetune.io import read_jsonl


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_split_markdown_keeps_code_fence_and_heading() -> None:
    markdown = """---
title: Demo
---
# Troubleshooting

This is a sufficiently long explanation about a Kubernetes failure and its diagnosis.

```yaml
apiVersion: v1
kind: Pod
```
"""
    chunks = split_markdown(markdown, min_chars=40, max_chars=1000, overlap_chars=20)
    assert len(chunks) == 1
    assert chunks[0][0] == "Troubleshooting"
    assert "```yaml" in chunks[0][1]


def test_split_markdown_enforces_max_for_long_paragraph() -> None:
    markdown = "# Long section\n\n" + ("diagnostic text " * 300)
    chunks = split_markdown(markdown, min_chars=40, max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    assert all(40 <= len(chunk) <= 500 for _, chunk in chunks)


def test_git_extractor_is_idempotent_and_records_revision(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-b", "main")
    _git(origin, "config", "user.email", "test@example.invalid")
    _git(origin, "config", "user.name", "Test")
    docs = origin / "content"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Docker build failure\n\n" + "Troubleshoot a Docker build timeout using logs and cache inspection. " * 8,
        encoding="utf-8",
    )
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "docs")

    output = tmp_path / "out" / "docs.jsonl"
    config = {
        "project": {"raw_root": str(tmp_path / "raw")},
        "defaults": {
            "output_template": str(output).replace("\\", "/"),
            "idempotency": {"skip_existing": True, "manifest_suffix": ".manifest.json"},
            "documentation_chunking": {"min_chars": 100, "max_chars": 1200, "overlap_chars": 50},
        },
        "category_rules": {
            "priority": ["troubleshooting", "concepto"],
            "troubleshooting": {"keywords": ["troubleshoot", "failure"]},
            "concepto": {"keywords": []},
            "fallback": "concepto",
        },
    }
    source = {
        "kind": "git_markdown",
        "tier": 1,
        "repository": str(origin),
        "ref": "main",
        "include": ["content/**/*.md"],
        "license": "Apache-2.0",
        "license_url": "https://example.invalid/license",
        "normalization": "reverse_instruction",
    }
    first = GitMarkdownExtractor(source_name="docs", source_config=source, root_config=config).extract()
    second = GitMarkdownExtractor(source_name="docs", source_config=source, root_config=config).extract()

    assert first.records == 1
    assert first.skipped is False
    assert second.skipped is True
    record = list(read_jsonl(output))[0]
    assert len(record["source_revision"]) == 40
    assert record["category"] == "troubleshooting"
    assert record["license"] == "Apache-2.0"
