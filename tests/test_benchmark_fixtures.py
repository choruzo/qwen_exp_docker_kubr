from __future__ import annotations

from pathlib import Path

from docker_k8s_finetune.io import read_jsonl


def test_out_of_domain_fixture_is_frozen_and_unique() -> None:
    records = list(read_jsonl(Path("benchmarks/fixtures/out_of_domain.jsonl")))
    assert len(records) == 50
    identifiers = {record["meta"]["content_hash"] for record in records}
    assert len(identifiers) == 50
    assert {record["meta"]["category"] for record in records} == {"out_of_domain"}
