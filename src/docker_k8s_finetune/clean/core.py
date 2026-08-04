from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect_langs


@dataclass(frozen=True)
class CleanResult:
    record: dict[str, Any] | None
    reason: str | None
    redactions: dict[str, int]


def html_to_text(value: str) -> str:
    """Strip HTML while retaining inline code and fenced preformatted blocks."""
    if not re.search(r"</?[A-Za-z][^>]*>", value):
        return value
    soup = BeautifulSoup(value, "html.parser")
    for node in soup.find_all("pre"):
        code = node.get_text("", strip=False).strip("\n")
        node.replace_with(f"\n```\n{code}\n```\n")
    for node in soup.find_all("code"):
        node.replace_with(f"`{node.get_text('', strip=False)}`")
    return soup.get_text("\n")


def repair_fences(value: str) -> str:
    value = re.sub(r"(?m)^\s*```\s*```\s*$", "", value)
    if len(re.findall(r"(?m)^\s*```", value)) % 2:
        value = value.rstrip() + "\n```"
    return value


def normalize_text(value: str, boilerplate_patterns: list[str]) -> str:
    value = html_to_text(value).replace("\r\n", "\n").replace("\r", "\n")
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in boilerplate_patterns]
    lines = [line.rstrip() for line in value.splitlines()]
    lines = [line for line in lines if not any(pattern.match(line) for pattern in patterns)]
    value = "\n".join(lines)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return repair_fences(value)


def redact_pii(value: str, patterns: Mapping[str, str]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for kind, pattern in patterns.items():
        value, count = re.subn(pattern, f"<REDACTED_{kind.upper()}>", value)
        if count:
            counts[kind] = count
    return value, counts


def detected_language(value: str, *, seed: int, minimum_letters: int) -> tuple[str | None, float]:
    if sum(character.isalpha() for character in value) < minimum_letters:
        return None, 0.0
    DetectorFactory.seed = seed
    try:
        candidates = detect_langs(value)
    except LangDetectException:
        return None, 0.0
    if not candidates:
        return None, 0.0
    return candidates[0].lang, float(candidates[0].prob)


def clean_record(record: Mapping[str, Any], config: Mapping[str, Any]) -> CleanResult:
    cleaned = deepcopy(dict(record))
    patterns = list(config.get("boilerplate", {}).get("line_patterns", []))
    pii_patterns = config["pii"]["patterns"]
    redactions: dict[str, int] = {}

    content = normalize_text(str(cleaned.get("raw_content", "")), patterns)
    content, content_redactions = redact_pii(content, pii_patterns)
    redactions.update(content_redactions)

    pair = cleaned.get("metadata", {}).get("pair")
    if isinstance(pair, dict):
        clean_pair: dict[str, str] = {}
        for field in ("system", "user", "assistant"):
            if pair.get(field) is None:
                continue
            text = normalize_text(str(pair[field]), patterns)
            text, counts = redact_pii(text, pii_patterns)
            clean_pair[field] = text
            for kind, count in counts.items():
                redactions[kind] = redactions.get(kind, 0) + count
        cleaned.setdefault("metadata", {})["pair"] = clean_pair

    limits = config["length"]
    if len(content) < int(limits["min_chars"]):
        return CleanResult(None, "too_short", redactions)
    if len(content) > int(limits["max_chars"]):
        return CleanResult(None, "too_long", redactions)

    language_config = config["language"]
    source = str(cleaned.get("source", ""))
    category = str(cleaned.get("category", ""))
    if source in language_config.get("trusted_sources", []):
        language, probability = str(cleaned.get("language") or "en"), 1.0
    elif category in language_config.get("code_categories", []):
        language, probability = "code", 1.0
    else:
        language, probability = detected_language(
            content,
            seed=int(config["seed"]),
            minimum_letters=int(language_config["minimum_letters"]),
        )
        if language not in language_config["allowed"] or probability < float(language_config["min_probability"]):
            return CleanResult(None, "language_not_allowed_or_uncertain", redactions)

    cleaned["raw_content"] = content
    cleaned["language"] = language
    metadata = cleaned.setdefault("metadata", {})
    metadata["cleaning"] = {
        "language_probability": round(probability, 6),
        "pii_redactions": redactions,
        "version": int(config["version"]),
    }
    return CleanResult(cleaned, None, redactions)
