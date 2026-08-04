from __future__ import annotations

import heapq
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import requests

from ..config import dotted_get
from ..errors import ExtractionError
from ..io import atomic_write_jsonl, content_hash
from ..schema import InterimRecord
from .base import BaseExtractor, ExtractionResult, first_present, infer_category
from .git_markdown import run_git


LOGGER = logging.getLogger(__name__)

SPDX_ALIASES = {
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "zlib": "Zlib",
    "unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
}


def normalize_licenses(value: Any) -> list[str]:
    if value is None:
        return []
    values: list[Any]
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    normalized: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            nested = item.get("license") or item.get("spdx_id") or item.get("name")
            normalized.extend(normalize_licenses(nested))
            continue
        text = str(item).strip()
        if not text:
            continue
        normalized.append(SPDX_ALIASES.get(text.casefold(), text))
    return list(dict.fromkeys(normalized))


class HuggingFaceExtractor(BaseExtractor):
    def _load_dataset(self) -> tuple[Iterable[Mapping[str, Any]], str | None]:
        try:
            from datasets import load_dataset
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ExtractionError("Hugging Face extraction dependencies are not installed") from exc

        if self.source_config.get("git_lfs_data_file"):
            return self._load_git_lfs_parquet(load_dataset)

        dataset_name = str(self.source_config["dataset"])
        revision = str(self.source_config.get("revision", "main"))
        resolved_revision: str | None = revision
        try:
            resolved_revision = HfApi().dataset_info(dataset_name, revision=revision).sha
        except Exception as exc:
            LOGGER.warning("Could not resolve immutable dataset revision for %s: %s", dataset_name, exc)

        kwargs: dict[str, Any] = {
            "path": dataset_name,
            "split": self.source_config.get("split", "train"),
            "revision": resolved_revision or revision,
            "streaming": bool(self.source_config.get("streaming", False)),
            "cache_dir": str(Path(str(self.project.get("raw_root", "data/raw"))) / "hf_cache"),
        }
        if self.source_config.get("subset"):
            kwargs["name"] = self.source_config["subset"]
        try:
            dataset = load_dataset(**kwargs)
        except Exception as exc:
            if self.source_config.get("viewer_fallback"):
                LOGGER.warning("Native dataset load failed for %s; using Dataset Viewer rows: %s", dataset_name, exc)
                dataset = self._viewer_rows()
            else:
                raise ExtractionError(f"Could not load Hugging Face dataset {dataset_name}: {exc}") from exc
        return dataset, resolved_revision

    def _load_git_lfs_parquet(self, load_dataset: Any) -> tuple[Iterable[Mapping[str, Any]], str]:
        dataset_name = str(self.source_config["dataset"])
        ref = str(self.source_config.get("revision", "main"))
        raw_root = Path(str(self.project.get("raw_root", "data/raw")))
        repository_dir = raw_root / "huggingface" / self.source_name
        repository_url = f"https://huggingface.co/datasets/{dataset_name}"
        if not (repository_dir / ".git").is_dir():
            repository_dir.parent.mkdir(parents=True, exist_ok=True)
            run_git(["clone", "--depth", "1", "--branch", ref, repository_url, str(repository_dir)])
        elif self.force:
            run_git(["fetch", "origin", ref, "--depth", "1"], cwd=repository_dir)
            run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=repository_dir)
            run_git(["lfs", "pull"], cwd=repository_dir)
        revision = run_git(["rev-parse", "HEAD"], cwd=repository_dir)
        data_file = repository_dir / str(self.source_config["git_lfs_data_file"])
        if not data_file.is_file():
            raise ExtractionError(f"Configured Git LFS data file does not exist: {data_file}")
        with data_file.open("rb") as handle:
            first = handle.read(4)
            handle.seek(-4, 2)
            last = handle.read(4)
        if first != b"PAR1" or last != b"PAR1":
            raise ExtractionError(f"Configured file is not a complete Parquet artifact: {data_file}")
        expected_sha256 = self.source_config.get("expected_data_sha256")
        if expected_sha256:
            import hashlib

            digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
            if digest != str(expected_sha256):
                raise ExtractionError(f"Parquet SHA-256 mismatch for {data_file}")
        try:
            dataset = load_dataset(
                "parquet",
                data_files=str(data_file),
                split=self.source_config.get("split", "train"),
                cache_dir=str(raw_root / "hf_cache"),
            )
        except Exception as exc:
            raise ExtractionError(f"Could not read verified Parquet {data_file}: {exc}") from exc
        return dataset, revision

    def _viewer_rows(self) -> Iterator[Mapping[str, Any]]:
        endpoint = "https://datasets-server.huggingface.co/rows"
        length = 100
        offset = 0
        maximum = self.source_config.get("max_records")
        while maximum is None or offset < int(maximum):
            response = requests.get(
                endpoint,
                params={
                    "dataset": self.source_config["dataset"],
                    "config": self.source_config.get("viewer_config", "default"),
                    "split": self.source_config.get("split", "train"),
                    "offset": offset,
                    "length": min(length, int(maximum) - offset) if maximum is not None else length,
                },
                timeout=60,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as viewer_exc:
                raise ExtractionError(f"Dataset Viewer rows request failed at offset {offset}") from viewer_exc
            payload = response.json()
            rows = payload.get("rows", [])
            if not rows:
                return
            for item in rows:
                row = item.get("row")
                if isinstance(row, Mapping):
                    yield row
            offset += len(rows)
            total = payload.get("num_rows_total")
            if total is not None and offset >= int(total):
                return

    def _row_license(self, row: Mapping[str, Any]) -> str | None:
        configured = str(self.source_config.get("license", "unknown"))
        if configured != "per_record":
            return configured
        licenses: list[str] = []
        for field in self.source_config.get("license_fields", []):
            licenses.extend(normalize_licenses(row.get(str(field))))
        allowed = set(dotted_get(self.root_config, str(self.source_config["per_record_allowlist_ref"])))
        return next((license_name for license_name in licenses if license_name in allowed), None)

    def _direct_pair(self, row: Mapping[str, Any]) -> tuple[str, str] | None:
        fields = self.source_config.get("field_map", {})
        if "user" in fields and "assistant" in fields:
            user = row.get(str(fields["user"]))
            assistant = row.get(str(fields["assistant"]))
        elif "user_candidates" in fields:
            user = first_present(row, fields["user_candidates"])
            assistant = first_present(row, fields.get("assistant_candidates", []))
        else:
            title = first_present(row, fields.get("question_candidates", []))
            body = row.get("body") or row.get("question_body")
            user = f"{title}\n\n{body}".strip() if title and body and str(body) not in str(title) else title
            assistant = first_present(row, fields.get("answer_candidates", []))
        if user is None or assistant is None:
            return None
        return str(user).strip(), str(assistant).strip()

    def _source_url(self, row: Mapping[str, Any], row_index: int) -> str:
        field_map = self.source_config.get("field_map", {})
        candidate = first_present(row, field_map.get("url_candidates", []))
        if candidate:
            return str(candidate)
        dataset = self.source_config["dataset"]
        return f"https://huggingface.co/datasets/{dataset}?row={row_index}"

    def _convert_row(self, row: Mapping[str, Any], row_index: int, revision: str | None) -> InterimRecord | None:
        normalization = self.source_config.get("normalization")
        license_name = self._row_license(row)
        if not license_name:
            return None
        metadata: dict[str, Any] = {"normalization": normalization, "dataset_row": row_index}
        if normalization == "direct_pair":
            pair = self._direct_pair(row)
            if not pair:
                return None
            user, assistant = pair
            raw_content = f"USER:\n{user}\n\nASSISTANT:\n{assistant}"
            metadata["pair"] = {"user": user, "assistant": assistant}
        else:
            content_fields = self.source_config.get("content_fields")
            if content_fields:
                payload = {str(field): row.get(str(field)) for field in content_fields}
                if not any(value not in (None, "", []) for value in payload.values()):
                    return None
                raw_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            else:
                field = str(self.source_config.get("content_field", "content"))
                content = row.get(field)
                if content is None:
                    return None
                raw_content = str(content).strip()
            if not raw_content or len(raw_content) > int(self.source_config.get("max_raw_chars", 50000)):
                return None

        source_url = self._source_url(row, row_index)
        category = str(self.source_config.get("category") or infer_category(raw_content, self.root_config))
        record_id = content_hash(f"{self.source_name}:{revision}:{row_index}:{raw_content}")
        field_map = self.source_config.get("field_map", {})
        author = first_present(row, field_map.get("author_candidates", []))
        author = author or row.get("display_name") or row.get("author")
        return InterimRecord(
            source=self.source_name,
            category=category,
            raw_content=raw_content,
            license=license_name,
            license_url=self.source_config.get("license_url"),
            url=source_url,
            source_record_id=record_id,
            source_revision=revision,
            source_path=str(row_index),
            title=str(row.get("title")) if row.get("title") else None,
            author=str(author) if author else None,
            attribution=f"{self.source_config['dataset']} row {row_index}",
            language=str(row.get("language") or "en"),
            metadata=metadata,
        )

    def _selected_records(
        self, dataset: Iterable[Mapping[str, Any]], revision: str | None
    ) -> tuple[list[InterimRecord], list[dict[str, Any]]]:
        maximum = self.source_config.get("sampling", {}).get("max_records", self.source_config.get("max_records"))
        max_records = int(maximum) if maximum is not None else None
        deterministic_hash_sampling = self.source_config.get("sampling", {}).get("strategy") == "deterministic_hash"
        selected: list[InterimRecord] = []
        rejected: list[dict[str, Any]] = []
        heap: list[tuple[int, int, InterimRecord]] = []
        for row_index, row in enumerate(dataset):
            record = self._convert_row(row, row_index, revision)
            if record is None:
                if str(self.source_config.get("license")) == "per_record":
                    rejected.append({
                        "source": self.source_name,
                        "source_record_id": str(row_index),
                        "license": "unknown_or_disallowed",
                        "url": self._source_url(row, row_index),
                        "reason": "per_record_license_not_allowlisted_or_invalid_content",
                    })
                continue
            if deterministic_hash_sampling and max_records is not None:
                key = int(content_hash(record.source_record_id or str(row_index)), 16)
                item = (-key, row_index, record)
                if len(heap) < max_records:
                    heapq.heappush(heap, item)
                elif key < -heap[0][0]:
                    heapq.heapreplace(heap, item)
                continue
            selected.append(record)
            if max_records is not None and len(selected) >= max_records:
                break
        if heap:
            selected = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        return selected, rejected

    def extract(self) -> ExtractionResult:
        if self.should_skip():
            return self.skipped_result()
        dataset, revision = self._load_dataset()
        selected, rejected = self._selected_records(dataset, revision)
        quarantine_path = Path(str(self.project.get("quarantine_root", "data/quarantine"))) / f"{self.source_name}_rejected.jsonl"
        if rejected:
            atomic_write_jsonl(quarantine_path, rejected)
        return self.write_records(
            selected,
            source_revision=revision,
            quarantined=len(rejected),
            extra_manifest={"quarantine_output": str(quarantine_path) if rejected else None},
        )
