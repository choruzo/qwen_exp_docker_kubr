from __future__ import annotations

import pytest

from docker_k8s_finetune.normalize.core import direct_pair_to_chatml
from docker_k8s_finetune.normalize.generate import _endpoint, parse_generated_user


def record(pair):
    return {
        "source": "unit",
        "category": "comando_cli",
        "license": "Apache-2.0",
        "url": "https://example.test/1",
        "source_record_id": "one",
        "metadata": {"pair": pair},
    }


def test_direct_pair_to_chatml() -> None:
    value = direct_pair_to_chatml(record({"user": "List pods", "assistant": "kubectl get pods"}), "system")
    assert [message["role"] for message in value["messages"]] == ["system", "user", "assistant"]
    assert value["messages"][1]["content"] == "List pods"
    assert value["meta"]["normalization"] == "direct_pair"


def test_direct_pair_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="empty"):
        direct_pair_to_chatml(record({"user": "", "assistant": "answer"}), "system")


def test_parse_generated_user_accepts_fenced_json() -> None:
    assert parse_generated_user('```json\n{"user": "Create a Pod"}\n```') == "Create a Pod"


def test_openai_compatible_endpoint() -> None:
    assert _endpoint("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"
