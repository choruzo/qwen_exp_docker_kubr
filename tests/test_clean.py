from __future__ import annotations

from docker_k8s_finetune.clean.core import clean_record, html_to_text, repair_fences


CONFIG = {
    "version": 1,
    "seed": 3407,
    "length": {"min_chars": 10, "max_chars": 50000},
    "language": {
        "allowed": ["en", "es"],
        "minimum_letters": 20,
        "min_probability": 0.5,
        "trusted_sources": ["docs"],
        "code_categories": ["comando_cli"],
    },
    "boilerplate": {"line_patterns": [r"^edit this page$"]},
    "pii": {
        "patterns": {
            "email": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "private_ipv4": r"\b192\.168(?:\.\d{1,3}){2}\b",
        }
    },
}


def base_record(**updates):
    record = {
        "source": "docs",
        "category": "concepto",
        "raw_content": "A sufficiently long technical explanation about Kubernetes workloads.",
        "license": "Apache-2.0",
        "url": "https://example.test/doc",
        "metadata": {},
    }
    record.update(updates)
    return record


def test_html_preserves_code() -> None:
    value = html_to_text("<p>Run <code>kubectl get pods</code>.</p><pre>apiVersion: v1</pre>")
    assert "`kubectl get pods`" in value
    assert "```\napiVersion: v1\n```" in value


def test_repairs_unclosed_fence() -> None:
    assert repair_fences("text\n```\ncode").endswith("\n```")


def test_clean_record_redacts_pii_and_boilerplate() -> None:
    result = clean_record(base_record(raw_content=(
        "Edit this page\nContact admin@example.test and connect to 192.168.1.20. "
        "This contains enough explanatory English text for a trusted documentation record."
    )), CONFIG)
    assert result.reason is None
    assert result.record is not None
    assert "admin@example.test" not in result.record["raw_content"]
    assert "<REDACTED_EMAIL>" in result.record["raw_content"]
    assert "<REDACTED_PRIVATE_IPV4>" in result.record["raw_content"]
    assert "Edit this page" not in result.record["raw_content"]
    assert result.redactions == {"email": 1, "private_ipv4": 1}


def test_pair_is_cleaned_independently() -> None:
    record = base_record(metadata={"pair": {
        "user": "Please email ops@example.test about this Kubernetes deployment question.",
        "assistant": "Use kubectl against 192.168.2.4 and inspect the workload carefully.",
    }})
    result = clean_record(record, CONFIG)
    assert result.record is not None
    pair = result.record["metadata"]["pair"]
    assert pair["user"].startswith("Please email <REDACTED_EMAIL>")
    assert "<REDACTED_PRIVATE_IPV4>" in pair["assistant"]
