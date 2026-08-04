from __future__ import annotations

from typing import Any, Mapping

from ..schema import ChatRecord


def direct_pair_to_chatml(record: Mapping[str, Any], system_prompt: str) -> dict[str, Any]:
    pair = record.get("metadata", {}).get("pair")
    if not isinstance(pair, Mapping):
        raise ValueError("record does not contain metadata.pair")
    user = str(pair.get("user", "")).strip()
    assistant = str(pair.get("assistant", "")).strip()
    if not user or not assistant:
        raise ValueError("direct pair has an empty user or assistant message")
    system = str(pair.get("system") or system_prompt).strip()
    meta = {
        "source": record.get("source"),
        "category": record.get("category"),
        "license": record.get("license"),
        "url": record.get("url"),
        "attribution": record.get("attribution"),
        "license_url": record.get("license_url"),
        "source_record_id": record.get("source_record_id"),
        "source_revision": record.get("source_revision"),
        "language": record.get("language"),
        "normalization": "direct_pair",
    }
    meta = {key: value for key, value in meta.items() if value is not None}
    chat = ChatRecord(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        meta=meta,
    )
    return chat.to_dict()


def reverse_pair_to_chatml(
    candidate: Mapping[str, Any], user: str, system_prompt: str, generator_model: str
) -> dict[str, Any]:
    meta = {
        "source": candidate.get("source"),
        "category": candidate.get("category"),
        "license": candidate.get("license"),
        "url": candidate.get("url"),
        "attribution": candidate.get("attribution"),
        "source_record_id": candidate.get("candidate_id"),
        "normalization": "reverse_instruction",
        "generator_model": generator_model,
    }
    meta = {key: value for key, value in meta.items() if value is not None}
    chat = ChatRecord(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": str(candidate["assistant_reference"]).strip()},
        ],
        meta=meta,
    )
    return chat.to_dict()
