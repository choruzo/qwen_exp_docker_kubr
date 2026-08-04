from __future__ import annotations

from docker_k8s_finetune.validate.core import extract_dockerfile, extract_yaml, fenced_blocks


def test_extract_yaml_from_fence() -> None:
    assert extract_yaml("answer\n```yaml\napiVersion: v1\nkind: Pod\n```") == "apiVersion: v1\nkind: Pod"


def test_extract_dockerfile_from_fence() -> None:
    assert extract_dockerfile("```dockerfile\nFROM python:3.12\nRUN true\n```").startswith("FROM python")


def test_unrelated_fence_is_not_yaml() -> None:
    assert extract_yaml("```bash\nkubectl get pods\n```") is None
