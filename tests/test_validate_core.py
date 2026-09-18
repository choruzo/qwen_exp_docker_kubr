from __future__ import annotations

from docker_k8s_finetune.validate.core import extract_dockerfile, extract_yaml, fenced_blocks
from pathlib import Path

from docker_k8s_finetune.validate import pipeline
from docker_k8s_finetune.validate.pipeline import _hadolint_issue_fails


def test_extract_yaml_from_fence() -> None:
    assert extract_yaml("answer\n```yaml\napiVersion: v1\nkind: Pod\n```") == "apiVersion: v1\nkind: Pod"


def test_extract_dockerfile_from_fence() -> None:
    assert extract_dockerfile("```dockerfile\nFROM python:3.12\nRUN true\n```").startswith("FROM python")


def test_extract_dockerfile_from_unlabelled_fence() -> None:
    assert extract_dockerfile("answer\n```\nFROM python:3.12\nCOPY . /app\n```").startswith("FROM python")


def test_plain_sentence_starting_with_from_is_not_a_dockerfile() -> None:
    assert extract_dockerfile("From inside a container, connect to the host with host.docker.internal.") is None


def test_hadolint_threshold_filters_lower_severity_diagnostics() -> None:
    assert _hadolint_issue_fails({"level": "error"}, "error") is True
    assert _hadolint_issue_fails({"level": "warning"}, "error") is False
    assert _hadolint_issue_fails({"level": "warning"}, "warning") is True


def test_hadolint_runs_large_file_sets_in_batches(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]):
        calls.append(command)
        return pipeline.subprocess.CompletedProcess(command, 0, "[]", "")

    monkeypatch.setattr(pipeline, "_run", fake_run)
    files = [tmp_path / f"{index}.Dockerfile" for index in range(5)]
    config = {
        "hadolint": {
            "image": "hadolint:test",
            "failure_threshold": "error",
            "batch_size": 2,
        }
    }

    assert pipeline._validate_dockerfiles(tmp_path, files, config) == {}
    assert [len(call[11:]) for call in calls] == [2, 2, 1]


def test_unrelated_fence_is_not_yaml() -> None:
    assert extract_yaml("```bash\nkubectl get pods\n```") is None
