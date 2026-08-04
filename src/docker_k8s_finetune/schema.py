from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .config import REQUIRED_CATEGORIES
from .errors import PipelineError


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class InterimRecord:
    source: str
    category: str
    raw_content: str
    license: str
    url: str
    retrieved_at: str = field(default_factory=utc_now_iso)
    source_record_id: str | None = None
    source_revision: str | None = None
    source_path: str | None = None
    title: str | None = None
    author: str | None = None
    attribution: str | None = None
    license_url: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for name in ("source", "raw_content", "license", "url", "retrieved_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PipelineError(f"Interim record field {name!r} must be a non-empty string")
        if self.category not in REQUIRED_CATEGORIES:
            raise PipelineError(f"Unsupported category: {self.category}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InterimRecord":
        record = cls(**dict(value))
        record.validate()
        return record


@dataclass
class ChatRecord:
    messages: list[dict[str, str]]
    meta: dict[str, Any]

    def validate(self) -> None:
        roles = [message.get("role") for message in self.messages]
        if roles != ["system", "user", "assistant"]:
            raise PipelineError(f"ChatML roles must be system/user/assistant, got {roles}")
        if any(not str(message.get("content", "")).strip() for message in self.messages):
            raise PipelineError("ChatML message content cannot be empty")
        if self.meta.get("category") not in REQUIRED_CATEGORIES:
            raise PipelineError("ChatML record has an invalid category")
        for required in ("source", "category", "license", "url"):
            if not self.meta.get(required):
                raise PipelineError(f"ChatML metadata is missing {required}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"messages": self.messages, "meta": self.meta}

