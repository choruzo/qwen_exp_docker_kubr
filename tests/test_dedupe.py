from __future__ import annotations

from docker_k8s_finetune.dedupe.exact import canonical_content


def chat(user: str, assistant: str, system: str = "system"):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def test_canonical_content_ignores_system_prompt() -> None:
    assert canonical_content(chat("List pods", "kubectl get pods", "one")) == canonical_content(
        chat("List pods", "kubectl get pods", "two")
    )


def test_canonical_content_normalizes_newlines_and_trailing_spaces() -> None:
    assert canonical_content(chat("List pods  \r\nnow", "ok")) == canonical_content(
        chat("List pods\nnow", "ok")
    )
