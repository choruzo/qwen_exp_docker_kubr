from __future__ import annotations

import json

import pytest

from docker_k8s_finetune.normalize.core import direct_pair_to_chatml
from docker_k8s_finetune.normalize import generate as generate_module
from docker_k8s_finetune.normalize.generate import (
    GeneratedInstructionRejected,
    _endpoint,
    _ordered_parallel_map,
    _read_generation_cache,
    generate_user,
    parse_generated_user,
)


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


@pytest.mark.parametrize("instruction", [
    "Explica el fragmento proporcionado.",
    "Resume la documentación proporcionada.",
    "¿Qué se recomienda según el documento?",
    "¿Qué componente aparece en el fragmento de código proporcionado?",
    "Resume el proyecto mencionado en el documento.",
    "Escribe una instrucción que solicite crear un Pod.",
])
def test_parse_generated_user_rejects_non_self_contained_instruction(instruction: str) -> None:
    with pytest.raises(ValueError, match="not self-contained"):
        parse_generated_user(json.dumps({"user": instruction}))


def test_openai_compatible_endpoint() -> None:
    assert _endpoint("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"


def test_ordered_parallel_map_preserves_order() -> None:
    assert list(_ordered_parallel_map(lambda value: value * 2, range(20), max_workers=3)) == [
        value * 2 for value in range(20)
    ]


def test_deterministic_quality_failure_has_distinct_exception(monkeypatch) -> None:
    monkeypatch.setenv("TEST_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("TEST_MODEL", "test-model")
    monkeypatch.setattr(generate_module.time, "sleep", lambda seconds: None)
    config = {
        "seed": 7,
        "reverse_instruction": {
            "base_url_env": "TEST_BASE_URL",
            "model_env": "TEST_MODEL",
            "api_key_env": "TEST_API_KEY",
            "prompt": "Create a self-contained instruction",
            "max_reference_chars": 1000,
            "max_retries": 2,
            "timeout_seconds": 1,
        },
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"user":"Resume el documento proporcionado."}'}}]}

    class FakeSession:
        calls = 0
        payloads = []

        def post(self, *args, **kwargs):
            self.calls += 1
            self.payloads.append(kwargs["json"])
            return FakeResponse()

    session = FakeSession()
    candidate = {"assistant_reference": "reference", "category": "concepto"}
    with pytest.raises(GeneratedInstructionRejected, match="not self-contained"):
        generate_user(candidate, config, session=session)
    assert session.calls == 2
    assert all(payload["max_tokens"] == 512 for payload in session.payloads)


def test_corrupt_generation_cache_is_ignored(tmp_path) -> None:
    cache = tmp_path / "candidate.json"
    cache.write_text('{"fingerprint":', encoding="utf-8")
    assert _read_generation_cache(cache, "expected") is None
    cache.write_text(json.dumps({"fingerprint": "other", "user": "value"}), encoding="utf-8")
    assert _read_generation_cache(cache, "expected") is None
