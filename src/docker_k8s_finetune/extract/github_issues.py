from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from typing import Any

import requests

from ..errors import ExtractionError
from ..io import content_hash
from ..schema import InterimRecord
from .base import BaseExtractor, ExtractionResult
from .huggingface import SPDX_ALIASES


class GitHubIssuesExtractor(BaseExtractor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        token_name = self.source_config.get("token_env", "GITHUB_TOKEN")
        token = os.environ.get(str(token_name))
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.api = str(self.source_config.get("api", "https://api.github.com")).rstrip("/")

    def _get(self, url: str, **params: Any) -> Any:
        response = self.session.get(url, params=params or None, timeout=60)
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            raise ExtractionError("GitHub API rate limit exhausted; configure GITHUB_TOKEN")
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ExtractionError(f"GitHub API request failed ({response.status_code}): {url}") from exc
        return response.json()

    def _license(self, repository: str) -> tuple[str | None, str | None]:
        payload = self._get(f"{self.api}/repos/{repository}/license")
        license_data = payload.get("license") or {}
        raw = str(license_data.get("spdx_id") or "").strip()
        normalized = SPDX_ALIASES.get(raw.casefold(), raw) if raw and raw != "NOASSERTION" else None
        return normalized, payload.get("html_url")

    def _iter_issues(self, repository: str) -> Iterator[Mapping[str, Any]]:
        page = 1
        while True:
            items = self._get(
                f"{self.api}/repos/{repository}/issues",
                state=self.source_config.get("state", "closed"),
                since=self.source_config.get("updated_since"),
                sort="updated",
                direction="desc",
                per_page=100,
                page=page,
            )
            if not items:
                return
            for issue in items:
                if "pull_request" not in issue:
                    yield issue
            if len(items) < 100:
                return
            page += 1

    def _resolution(self, issue: Mapping[str, Any]) -> Mapping[str, Any] | None:
        comments_url = issue.get("comments_url")
        if not comments_url:
            return None
        comments = self._get(str(comments_url), per_page=100)
        if not comments:
            return None
        trusted = {"OWNER", "MEMBER", "COLLABORATOR"}

        def score(comment: Mapping[str, Any]) -> tuple[int, int, str]:
            association = 1 if comment.get("author_association") in trusted else 0
            reactions = comment.get("reactions") or {}
            total = int(reactions.get("total_count", 0))
            return association, total, str(comment.get("created_at", ""))

        candidate = max(comments, key=score)
        minimum = int(self.source_config.get("minimum_resolution_reactions", 0))
        if score(candidate)[0] == 0 and score(candidate)[1] < minimum:
            return None
        return candidate

    def _records(self) -> Iterator[InterimRecord]:
        allowed = set(self.root_config.get("license_policy", {}).get("final_dataset_allowlist", []))
        excluded = re.compile(str(self.source_config.get("exclude_labels_regex", r"$^")))
        maximum = int(self.source_config.get("max_records_per_repository", 2500))
        for repository_config in self.source_config.get("repositories", []):
            repository = str(repository_config["name"])
            label_pattern = re.compile(str(repository_config.get("labels_any_regex", ".*")))
            license_name, license_url = self._license(repository)
            if license_name not in allowed:
                continue
            emitted = 0
            for issue in self._iter_issues(repository):
                labels = [str(label.get("name", "")) for label in issue.get("labels", [])]
                if any(excluded.search(label) for label in labels) or not any(label_pattern.search(label) for label in labels):
                    continue
                body = str(issue.get("body") or "").strip()
                if not body:
                    continue
                resolution = self._resolution(issue)
                if not resolution or not str(resolution.get("body") or "").strip():
                    continue
                question = f"{issue.get('title', '')}\n\n{body}".strip()
                answer = str(resolution["body"]).strip()
                url = str(issue["html_url"])
                yield InterimRecord(
                    source=self.source_name,
                    category="troubleshooting",
                    raw_content=f"USER:\n{question}\n\nASSISTANT:\n{answer}",
                    license=str(license_name),
                    license_url=license_url,
                    url=url,
                    source_record_id=f"{repository}#{issue['number']}",
                    source_revision=str(issue.get("updated_at")),
                    title=str(issue.get("title")),
                    author=str((issue.get("user") or {}).get("login") or "unknown"),
                    attribution=f"{repository} issue #{issue['number']} and resolving comment {resolution.get('html_url')}",
                    language="en",
                    metadata={
                        "normalization": "issue_resolution_pair",
                        "pair": {"user": question, "assistant": answer},
                        "labels": labels,
                        "resolution_url": resolution.get("html_url"),
                    },
                )
                emitted += 1
                if emitted >= maximum:
                    break

    def extract(self) -> ExtractionResult:
        if self.should_skip():
            return self.skipped_result()
        return self.write_records(self._records())

