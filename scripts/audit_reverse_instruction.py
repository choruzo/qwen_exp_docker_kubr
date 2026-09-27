#!/usr/bin/env python3
"""Audit reverse-instruction pairs for format and alignment risks without emitting dataset text.

Reverse-instruction records pair an LLM-generated question with a source document chunk used
verbatim as the answer. This script measures, per normalization/source/category group, the
signals that make such pairs teach open-ended enumeration: long bullet lists, raw doc-site
template residue, questions that cite context absent from the prompt, chunks that stop
inside a list/table, changelog-shaped answers, answer/question length imbalance and
question/answer language mismatch. Lexical alignment is reported, not flagged.

Only aggregates go to stdout. ``--flagged-output`` optionally writes content hashes and flag
names (no text) for later filtering; keep that file under the gitignored ``artifacts/``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


WORD_RE = re.compile(r"[\w.\-/]+", re.UNICODE)
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
RULE_RE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
ISSUE_REF_RE = re.compile(r"(?:\b[\w.-]+/[\w.-]+#\d+|\(#\d{2,}\)|\[#\d{2,}\])")
CHANGELOG_HEADING_RE = re.compile(
    r"^\s*#{1,6}\s*(?:v?\d+\.\d+|bug ?fixes|fixes|new\b|upgrades|updates|deprecat|removed|removals"
    r"|breaking|known issues|release notes|changelog|what'?s new|enhancements"
    r"|for (?:all platforms|windows|mac|linux))",
    re.IGNORECASE,
)
TEMPLATE_RE = re.compile(r"\{\{[<%]|[>%]\}\}|\]\(/(?:docs|[a-z]{2}/docs)/|\{\{\s*\.")
CONTEXT_REFERENCE_RE = re.compile(
    r"\b(?:seg[uú]n (?:la informaci[oó]n|el (?:texto|documento|fragmento|contenido|pasaje))"
    r"|(?:informaci[oó]n|texto|documento|fragmento|contenido) proporcionad[oa]"
    r"|(?:en|del) (?:el )?(?:fragmento|texto|documento|pasaje) (?:anterior|dado|siguiente)"
    r"|(?:mencionad[oa]s?|descrit[oa]s?|indicad[oa]s?) en (?:el|la) (?:texto|documento|fragmento|informaci[oó]n))",
    re.IGNORECASE,
)
# Questions that legitimately ask for a list; others are "narrow" questions.
ENUMERATION_CUE_RE = re.compile(
    r"\b(?:enumera|lista|listar|todos|todas|cu[aá]les son|qu[eé] (?:pasos|opciones|cambios|comandos|"
    r"requisitos|campos|tipos|funcionalidades|mejoras|correcciones)|pasos|list|all)\b",
    re.IGNORECASE,
)
SPANISH_STOPWORDS = frozenset(
    "el la los las de del que en y para por con una un es se como al lo su sus más pero "
    "este esta estos estas cuando puede pueden debe hay también sobre entre".split()
)
ENGLISH_STOPWORDS = frozenset(
    "the of and to in is for that with this are be as on it by can you an or from if "
    "when will which not your should must have".split()
)
STOPWORDS = SPANISH_STOPWORDS | ENGLISH_STOPWORDS | frozenset(
    "qué cuál cuáles cómo explica describe enumera indica genera muestra proporciona "
    "detalla lista todos todas incluyendo según información".split()
)


def role_text(record: dict, role: str) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in record.get("messages", [])
        if message.get("role") == role
    )


def _words(text: str) -> list[str]:
    return [word.strip(".-/").casefold() for word in WORD_RE.findall(text)]


def spanish_share(text: str) -> float | None:
    """Share of recognised stopwords that are Spanish; None when too few to decide."""
    words = _words(text)
    spanish = sum(word in SPANISH_STOPWORDS for word in words)
    english = sum(word in ENGLISH_STOPWORDS for word in words)
    if spanish + english < 5:
        return None
    return spanish / (spanish + english)


def question_term_recall(question: str, answer: str) -> float | None:
    """Fraction of distinctive question terms (identifiers, long words) present in the answer."""
    terms = {
        word for word in _words(question)
        if len(word) >= 4 and word not in STOPWORDS and not word.isdigit()
    }
    if not terms:
        return None
    answer_words = set(_words(answer))
    return sum(term in answer_words for term in terms) / len(terms)


def ends_in_structure(answer: str) -> bool:
    """True when the answer stops inside a list, table, heading, rule or template tag.

    Natural answers often end without punctuation (a URL, a command), so punctuation alone is
    not evidence of a cut chunk; ending on an open structure is.
    """
    lines = [line for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return True
    last = lines[-1]
    return any(pattern.search(last) for pattern in (
        BULLET_RE, TABLE_ROW_RE, HEADING_RE, RULE_RE, TEMPLATE_RE,
    )) or last.rstrip().endswith((":", ","))


def is_release_notes(answer: str) -> bool:
    """Changelog-shaped answer.

    Release notes are chunked per section, so a single changelog heading followed by a list
    is enough; several issue-referencing bullets also qualify without any heading.
    """
    lines = answer.splitlines()
    headings = sum(bool(CHANGELOG_HEADING_RE.match(line)) for line in lines)
    bullets = sum(bool(BULLET_RE.match(line)) for line in lines)
    issue_bullets = sum(
        bool(BULLET_RE.match(line)) and bool(ISSUE_REF_RE.search(line)) for line in lines
    )
    return issue_bullets >= 3 or (headings >= 1 and bullets >= 3)


def measure(record: dict) -> dict:
    question = role_text(record, "user")
    answer = role_text(record, "assistant")
    category = str(record.get("meta", {}).get("category", "unknown"))
    answer_lines = answer.splitlines()
    # YAML sequences look like bullets; only count markdown bullets outside YAML answers.
    bullets = 0 if category == "generacion_yaml" else sum(
        bool(BULLET_RE.match(line)) for line in answer_lines
    )
    question_words = max(len(question.split()), 1)
    is_yaml = category == "generacion_yaml"
    return {
        "question_words": len(question.split()),
        "answer_words": len(answer.split()),
        "length_ratio": len(answer.split()) / question_words,
        "bullets": bullets,
        "headings": sum(bool(HEADING_RE.match(line)) for line in answer_lines),
        "versions": len(VERSION_RE.findall(answer)),
        "template_residue": bool(TEMPLATE_RE.search(answer)),
        "context_reference": bool(CONTEXT_REFERENCE_RE.search(question)),
        "starts_with_heading": bool(answer_lines) and bool(HEADING_RE.match(answer_lines[0])),
        # YAML list items and English YAML comments are not chunk or language defects.
        "ends_in_structure": not is_yaml and ends_in_structure(answer),
        "release_notes": not is_yaml and is_release_notes(answer),
        "narrow_question": not ENUMERATION_CUE_RE.search(question),
        "question_spanish_share": None if is_yaml else spanish_share(question),
        "answer_spanish_share": None if is_yaml else spanish_share(answer),
        "term_recall": question_term_recall(question, answer),
    }


def flags(metrics: dict, thresholds: dict) -> list[str]:
    found = []
    if metrics["bullets"] >= thresholds["bullets"]:
        found.append("long_list")
    if metrics["template_residue"]:
        found.append("template_residue")
    if metrics["context_reference"]:
        found.append("context_reference")
    if metrics["ends_in_structure"]:
        found.append("ends_in_structure")
    if metrics["release_notes"]:
        found.append("release_notes")
    if metrics["narrow_question"] and metrics["bullets"] >= thresholds["narrow_list_bullets"]:
        found.append("narrow_question_list_answer")
    if metrics["length_ratio"] >= thresholds["length_ratio"]:
        found.append("length_imbalance")
    question_es, answer_es = metrics["question_spanish_share"], metrics["answer_spanish_share"]
    if question_es is not None and answer_es is not None and question_es >= 0.5 > answer_es:
        found.append("language_mismatch")
    if metrics["versions"] >= thresholds["versions"]:
        found.append("version_dense")
    return found


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p90": None}
    ordered = sorted(values)
    return {"median": median(ordered), "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}


def _summarize(rows: list[tuple[dict, list[str]]], thresholds: dict) -> dict:
    count = len(rows)
    flag_counts = Counter(flag for _, found in rows for flag in found)
    recalls = [metrics["term_recall"] for metrics, _ in rows if metrics["term_recall"] is not None]
    return {
        "records": count,
        "answer_words": _quantiles([metrics["answer_words"] for metrics, _ in rows]),
        "length_ratio": _quantiles([metrics["length_ratio"] for metrics, _ in rows]),
        "bullets": _quantiles([metrics["bullets"] for metrics, _ in rows]),
        # Informational only: natural answers rarely echo question terms, so low lexical
        # recall is not a defect by itself (direct pairs median ~0.14).
        "term_recall": _quantiles(recalls),
        "low_term_recall_rate": (
            sum(recall < thresholds["term_recall"] for recall in recalls) / len(recalls)
            if recalls else None
        ),
        "starts_with_heading_rate": sum(m["starts_with_heading"] for m, _ in rows) / count,
        "flag_rates": {flag: flag_counts[flag] / count for flag in sorted(flag_counts)},
        "any_flag_rate": sum(bool(found) for _, found in rows) / count,
        "two_or_more_flags_rate": sum(len(found) >= 2 for _, found in rows) / count,
    }


def audit(path: Path, *, thresholds: dict, flagged_output: Path | None = None) -> dict:
    by_normalization: dict[str, list] = defaultdict(list)
    by_group: dict[tuple[str, str, str], list] = defaultdict(list)
    flagged = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                meta = record["meta"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc
            metrics = measure(record)
            found = flags(metrics, thresholds)
            normalization = str(meta.get("normalization", "unknown"))
            by_normalization[normalization].append((metrics, found))
            key = (normalization, str(meta.get("source", "unknown")), str(meta.get("category", "unknown")))
            by_group[key].append((metrics, found))
            if found:
                flagged.append({
                    "content_hash": meta.get("content_hash"),
                    "normalization": normalization,
                    "source": key[1],
                    "category": key[2],
                    "flags": found,
                })
    if flagged_output is not None:
        flagged_output.parent.mkdir(parents=True, exist_ok=True)
        with flagged_output.open("w", encoding="utf-8") as handle:
            for row in flagged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "version": 1,
        "input": str(path),
        "thresholds": thresholds,
        "by_normalization": {
            name: _summarize(rows, thresholds) for name, rows in sorted(by_normalization.items())
        },
        "by_group": [
            {"normalization": normalization, "source": source, "category": category, **_summarize(rows, thresholds)}
            for (normalization, source, category), rows in sorted(by_group.items())
        ],
        "flagged_records": len(flagged),
        "flagged_output": str(flagged_output) if flagged_output else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("data/processed/train.jsonl"))
    parser.add_argument("--bullets", type=int, default=10)
    parser.add_argument("--length-ratio", type=float, default=15.0)
    parser.add_argument("--term-recall", type=float, default=0.3)
    parser.add_argument("--versions", type=int, default=10)
    parser.add_argument("--narrow-list-bullets", type=int, default=5)
    parser.add_argument("--flagged-output", type=Path)
    args = parser.parse_args()
    thresholds = {
        "bullets": args.bullets,
        "length_ratio": args.length_ratio,
        "term_recall": args.term_recall,
        "versions": args.versions,
        "narrow_list_bullets": args.narrow_list_bullets,
    }
    print(json.dumps(
        audit(args.path, thresholds=thresholds, flagged_output=args.flagged_output),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
