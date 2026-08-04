from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ..errors import ExtractionError
from ..io import content_hash
from ..schema import InterimRecord
from .base import BaseExtractor, ExtractionResult, infer_category


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def run_git(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise ExtractionError("git is required for documentation extraction") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "unknown git error").strip()
        raise ExtractionError(f"git {' '.join(args)} failed: {details}") from exc
    return completed.stdout.strip()


def strip_front_matter(markdown: str) -> str:
    return FRONT_MATTER_RE.sub("", markdown.replace("\r\n", "\n"), count=1)


def split_markdown(markdown: str, *, min_chars: int, max_chars: int, overlap_chars: int) -> list[tuple[str, str]]:
    """Split Markdown on headings and paragraph boundaries without breaking fences."""
    text = strip_front_matter(markdown)
    sections: list[tuple[str, str]] = []
    title = "Untitled"
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((title, body))
        buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        heading = HEADING_RE.match(line) if not in_fence else None
        if heading:
            flush()
            title = heading.group(2).strip()
            buffer.append(line)
        else:
            buffer.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for section_title, body in sections:
        if len(body) <= max_chars:
            if len(body) >= min_chars:
                chunks.append((section_title, body))
            continue
        paragraphs = re.split(r"\n{2,}", body)
        current = ""
        for raw_paragraph in paragraphs:
            raw_paragraph = raw_paragraph.strip()
            if not raw_paragraph:
                continue
            if len(raw_paragraph) > max_chars and ("```" in raw_paragraph or "~~~" in raw_paragraph):
                # A code fence cannot be split safely. Oversized code is noise for
                # the configured training context and is intentionally discarded.
                continue
            if len(raw_paragraph) > max_chars:
                stride = max(1, max_chars - overlap_chars)
                paragraph_pieces = [raw_paragraph[offset : offset + max_chars] for offset in range(0, len(raw_paragraph), stride)]
            else:
                paragraph_pieces = [raw_paragraph]

            for paragraph in paragraph_pieces:
                candidate = f"{current}\n\n{paragraph}".strip()
                if len(candidate) <= max_chars:
                    current = candidate
                    continue
                previous = current
                if len(previous) >= min_chars:
                    chunks.append((section_title, previous))
                overlap_budget = max(0, max_chars - len(paragraph) - 2)
                effective_overlap = min(overlap_chars, overlap_budget)
                overlap = previous[-effective_overlap:] if effective_overlap and previous else ""
                current = f"{overlap}\n\n{paragraph}".strip()
        if len(current) >= min_chars:
            chunks.append((section_title, current))
    if any(len(chunk) > max_chars for _, chunk in chunks):
        raise ValueError("Markdown chunker produced a chunk above max_chars")
    return chunks


class GitMarkdownExtractor(BaseExtractor):
    def _repository_dir(self) -> Path:
        raw_root = Path(str(self.project.get("raw_root", "data/raw")))
        return raw_root / "git" / self.source_name

    def _ensure_repository(self) -> tuple[Path, str]:
        repository = str(self.source_config["repository"])
        ref = str(self.source_config.get("ref", "main"))
        local_repository = Path(repository)
        if local_repository.is_dir() and (local_repository / ".git").is_dir():
            revision = run_git(["rev-parse", ref], cwd=local_repository)
            return local_repository.resolve(), revision

        destination = self._repository_dir()
        if not (destination / ".git").is_dir():
            if destination.exists() and any(destination.iterdir()):
                raise ExtractionError(f"Non-git extraction directory is not empty: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            run_git(["clone", "--depth", "1", "--branch", ref, repository, str(destination)])
        elif self.force:
            run_git(["fetch", "origin", ref, "--depth", "1"], cwd=destination)
            run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=destination)
        revision = run_git(["rev-parse", "HEAD"], cwd=destination)
        return destination, revision

    def _iter_paths(self, repository_dir: Path) -> list[Path]:
        includes = self.source_config.get("include", ["**/*.md"])
        excludes = self.source_config.get("exclude", [])
        found: dict[str, Path] = {}
        for pattern in includes:
            for path in repository_dir.glob(str(pattern)):
                if not path.is_file():
                    continue
                relative = path.relative_to(repository_dir).as_posix()
                if any(fnmatch.fnmatch(relative, str(exclude)) for exclude in excludes):
                    continue
                found[relative] = path
        return [found[key] for key in sorted(found)]

    def _record_url(self, relative: str, revision: str) -> str:
        repository = str(self.source_config["repository"]).removesuffix(".git")
        if repository.startswith("https://github.com/"):
            return f"{repository}/blob/{revision}/{relative}"
        return f"{repository}/src/commit/{revision}/{relative}"

    def _records(self, repository_dir: Path, revision: str) -> Iterable[InterimRecord]:
        chunking = {**self.defaults.get("documentation_chunking", {}), **self.source_config.get("chunking", {})}
        min_chars = int(chunking.get("min_chars", 300))
        max_chars = int(chunking.get("max_chars", 12000))
        overlap = int(chunking.get("overlap_chars", 300))
        maximum = self.source_config.get("max_records")
        emitted = 0
        for path in self._iter_paths(repository_dir):
            relative = path.relative_to(repository_dir).as_posix()
            try:
                markdown = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for chunk_index, (title, chunk) in enumerate(
                split_markdown(markdown, min_chars=min_chars, max_chars=max_chars, overlap_chars=overlap)
            ):
                category = self.source_config.get("category") or infer_category(chunk, self.root_config)
                record_id = content_hash(f"{revision}:{relative}:{chunk_index}:{chunk}")
                yield InterimRecord(
                    source=self.source_name,
                    category=str(category),
                    raw_content=chunk,
                    license=str(self.source_config["license"]),
                    license_url=self.source_config.get("license_url"),
                    url=self._record_url(relative, revision),
                    source_record_id=record_id,
                    source_revision=revision,
                    source_path=relative,
                    title=title,
                    attribution=f"{self.source_name}: {relative} @ {revision}",
                    language="en",
                    metadata={"chunk_index": chunk_index, "normalization": self.source_config.get("normalization")},
                )
                emitted += 1
                if maximum is not None and emitted >= int(maximum):
                    return

    def extract(self) -> ExtractionResult:
        if self.should_skip():
            return self.skipped_result()
        repository_dir, revision = self._ensure_repository()
        return self.write_records(self._records(repository_dir, revision), source_revision=revision)


class FailureStoryIndexExtractor(GitMarkdownExtractor):
    """Extract only the unlicensed index into quarantine; never linked pages."""

    def _ensure_repository(self) -> tuple[Path, str]:
        try:
            return super()._ensure_repository()
        except ExtractionError:
            fallback = self.source_config.get("fallback_repository")
            if not fallback:
                raise
            original_repo = self.source_config["repository"]
            original_ref = self.source_config.get("ref")
            self.source_config["repository"] = fallback
            self.source_config["ref"] = self.source_config.get("fallback_ref", "master")
            try:
                return super()._ensure_repository()
            finally:
                self.source_config["repository"] = original_repo
                self.source_config["ref"] = original_ref

    def _records(self, repository_dir: Path, revision: str) -> Iterable[InterimRecord]:
        for path in self._iter_paths(repository_dir):
            relative = path.relative_to(repository_dir).as_posix()
            content = path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            yield InterimRecord(
                source=self.source_name,
                category="troubleshooting",
                raw_content=content,
                license="unknown",
                url=self._record_url(relative, revision),
                source_record_id=content_hash(f"{revision}:{relative}"),
                source_revision=revision,
                source_path=relative,
                title=path.stem,
                attribution=f"Unlicensed index only: {relative} @ {revision}",
                metadata={"include_in_final": False, "requires_link_license_audit": True},
            )
