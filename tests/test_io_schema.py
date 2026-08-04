from pathlib import Path

import pytest

from docker_k8s_finetune.io import atomic_write_jsonl, read_jsonl
from docker_k8s_finetune.schema import ChatRecord, InterimRecord


def test_atomic_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    count, digest = atomic_write_jsonl(path, [{"value": "á"}, {"value": 2}])
    assert count == 2
    assert len(digest) == 64
    assert list(read_jsonl(path)) == [{"value": "á"}, {"value": 2}]
    assert not list(path.parent.glob("*.tmp"))


def test_interim_record_rejects_unknown_category() -> None:
    record = InterimRecord(
        source="test",
        category="other",
        raw_content="content",
        license="MIT",
        url="https://example.test",
    )
    with pytest.raises(Exception, match="Unsupported category"):
        record.to_dict()


def test_chatml_requires_exact_three_roles() -> None:
    record = ChatRecord(messages=[{"role": "user", "content": "hello"}], meta={})
    with pytest.raises(Exception, match="roles"):
        record.to_dict()

