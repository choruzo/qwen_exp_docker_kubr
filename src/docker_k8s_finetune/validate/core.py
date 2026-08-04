from __future__ import annotations

import re
from typing import Any, Mapping


def assistant_content(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                return content
    raise ValueError("record has no assistant content")


def fenced_blocks(text: str, languages: tuple[str, ...]) -> list[str]:
    language_pattern = "|".join(re.escape(language) for language in languages)
    pattern = re.compile(rf"```(?:{language_pattern})[ \t]*\n(.*?)```", re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text) if match.group(1).strip()]


def extract_yaml(text: str) -> str | None:
    blocks = fenced_blocks(text, ("yaml", "yml"))
    if blocks:
        return "\n---\n".join(blocks)
    stripped = text.strip()
    if re.search(r"(?m)^\s*apiVersion\s*:", stripped) and re.search(r"(?m)^\s*kind\s*:", stripped):
        return stripped
    return None


def extract_dockerfile(text: str) -> str | None:
    blocks = fenced_blocks(text, ("dockerfile", "docker"))
    if blocks:
        return "\n\n".join(blocks)
    stripped = text.strip()
    if re.search(r"(?mi)^\s*FROM\s+\S+", stripped):
        return stripped
    return None


def parse_json_output(value: str) -> list[dict[str, Any]]:
    import json

    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        results = []
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                results.append(item)
        return results
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return [parsed] if isinstance(parsed, dict) else []
