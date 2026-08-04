from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..io import atomic_write_json, atomic_write_jsonl, content_hash, stable_json
from ..schema import InterimRecord, utc_now_iso


LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    source: str
    output: str
    records: int
    skipped: bool
    sha256: str | None = None
    quarantined: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class BaseExtractor:
    def __init__(
        self,
        *,
        source_name: str,
        source_config: Mapping[str, Any],
        root_config: Mapping[str, Any],
        force: bool = False,
    ) -> None:
        self.source_name = source_name
        self.source_config = dict(source_config)
        self.root_config = root_config
        self.defaults = dict(root_config.get("defaults", {}))
        self.project = dict(root_config.get("project", {}))
        self.force = force

    @property
    def output_path(self) -> Path:
        explicit = self.source_config.get("destination")
        if explicit:
            return Path(str(explicit))
        template = str(self.defaults.get("output_template", "data/interim/extracted/{source}.jsonl"))
        return Path(template.format(source=self.source_name))

    @property
    def manifest_path(self) -> Path:
        suffix = self.defaults.get("idempotency", {}).get("manifest_suffix", ".manifest.json")
        return Path(str(self.output_path) + str(suffix))

    def should_skip(self) -> bool:
        skip_existing = self.defaults.get("idempotency", {}).get("skip_existing", True)
        return bool(skip_existing and not self.force and self.output_path.is_file() and self.manifest_path.is_file())

    def skipped_result(self) -> ExtractionResult:
        records = 0
        sha256 = None
        try:
            import json

            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            records = int(manifest.get("records", 0))
            sha256 = manifest.get("sha256")
        except (OSError, ValueError, TypeError):
            LOGGER.warning("Could not read existing manifest for %s", self.source_name)
        LOGGER.info("Skipping %s; output and manifest already exist", self.source_name)
        return ExtractionResult(self.source_name, str(self.output_path), records, True, sha256)

    def write_records(
        self,
        records: Iterable[InterimRecord | Mapping[str, Any]],
        *,
        source_revision: str | None = None,
        quarantined: int = 0,
        extra_manifest: Mapping[str, Any] | None = None,
    ) -> ExtractionResult:
        def serialized() -> Iterable[Mapping[str, Any]]:
            for record in records:
                yield record.to_dict() if isinstance(record, InterimRecord) else record

        count, sha256 = atomic_write_jsonl(self.output_path, serialized())
        manifest: dict[str, Any] = {
            "source": self.source_name,
            "kind": self.source_config.get("kind"),
            "output": str(self.output_path),
            "records": count,
            "quarantined": quarantined,
            "sha256": sha256,
            "source_revision": source_revision,
            "retrieved_at": utc_now_iso(),
            "source_config_sha256": content_hash(stable_json(self.source_config)),
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        atomic_write_json(self.manifest_path, manifest)
        LOGGER.info("Extracted %d records from %s", count, self.source_name)
        return ExtractionResult(
            source=self.source_name,
            output=str(self.output_path),
            records=count,
            skipped=False,
            sha256=sha256,
            quarantined=quarantined,
        )

    def extract(self) -> ExtractionResult:  # pragma: no cover - interface
        raise NotImplementedError


def infer_category(content: str, root_config: Mapping[str, Any], fallback: str | None = None) -> str:
    lowered = content.casefold()
    rules = root_config.get("category_rules", {})
    for category in rules.get("priority", []):
        keywords = rules.get(category, {}).get("keywords", [])
        if any(str(keyword).casefold() in lowered for keyword in keywords):
            return str(category)
    return fallback or str(rules.get("fallback", "concepto"))


def first_present(row: Mapping[str, Any], candidates: Iterable[str]) -> Any:
    for candidate in candidates:
        value = row.get(candidate)
        if value is not None and str(value).strip():
            return value
    return None

