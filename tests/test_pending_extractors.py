from datetime import datetime, timezone

import pytest

from docker_k8s_finetune.extract.github_issues import GitHubIssuesExtractor
from docker_k8s_finetune.extract.stackoverflow import QUERY, StackOverflowBigQueryExtractor


def test_github_resolution_paginates_all_comments(monkeypatch) -> None:
    extractor = GitHubIssuesExtractor(
        source_name="issues",
        source_config={"api": "https://api.github.test", "minimum_resolution_reactions": 1},
        root_config={},
    )
    calls = []

    def fake_get(url, **params):
        calls.append(params["page"])
        if params["page"] == 1:
            return [
                {"body": "noise", "author_association": "NONE", "reactions": {"total_count": 0}}
                for _ in range(100)
            ]
        return [{"body": "fixed", "author_association": "MEMBER", "reactions": {"total_count": 3}}]

    monkeypatch.setattr(extractor, "_get", fake_get)
    result = extractor._resolution({"comments_url": "https://api.github.test/comments", "comments": 101})
    assert calls == [1, 2]
    assert result["body"] == "fixed"


def test_github_resolution_skips_request_when_issue_has_no_comments(monkeypatch) -> None:
    extractor = GitHubIssuesExtractor(
        source_name="issues",
        source_config={"api": "https://api.github.test"},
        root_config={},
    )
    monkeypatch.setattr(extractor, "_get", lambda *args, **kwargs: pytest.fail("must not request comments"))
    assert extractor._resolution({"comments_url": "https://api.github.test/comments", "comments": 0}) is None


def test_github_search_splits_windows_over_api_result_cap(monkeypatch) -> None:
    extractor = GitHubIssuesExtractor(
        source_name="issues",
        source_config={"api": "https://api.github.test"},
        root_config={},
    )
    calls = []

    def fake_get(url, **params):
        calls.append(params["q"])
        if len(calls) == 1:
            return {"total_count": 1001, "incomplete_results": False, "items": []}
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [{"id": len(calls), "updated_at": "2024-01-01T00:00:00Z"}],
        }

    monkeypatch.setattr(extractor, "_get", fake_get)
    records = list(extractor._search_issue_window(
        "moby/moby",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
    ))
    assert [record["id"] for record in records] == [2, 3]
    assert len(calls) == 3
    assert "2024-01-02T00:00:01Z..2024-01-03T00:00:00Z" in calls[1]


def test_github_search_paginates_below_result_cap(monkeypatch) -> None:
    extractor = GitHubIssuesExtractor(
        source_name="issues",
        source_config={"api": "https://api.github.test"},
        root_config={},
    )
    pages = []

    def fake_get(url, **params):
        pages.append(params["page"])
        if params["page"] == 1:
            return {"total_count": 101, "incomplete_results": False, "items": [{"id": 1}] * 100}
        return {"total_count": 101, "incomplete_results": False, "items": [{"id": 2}]}

    monkeypatch.setattr(extractor, "_get", fake_get)
    records = list(extractor._search_issue_window(
        "moby/moby",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    ))
    assert len(records) == 101
    assert pages == [1, 2]


def test_github_recent_issue_listing_stops_before_search_fallback(monkeypatch) -> None:
    extractor = GitHubIssuesExtractor(
        source_name="issues",
        source_config={
            "api": "https://api.github.test",
            "updated_since": "2024-01-01T00:00:00Z",
        },
        root_config={},
    )
    pages = []

    def fake_get(url, **params):
        assert url.endswith("/repos/moby/moby/issues")
        pages.append(params["page"])
        if params["page"] == 1:
            return [{"id": value, "updated_at": "2024-01-02T00:00:00Z"} for value in range(100)]
        return [{"id": 100, "updated_at": "2024-01-01T12:00:00Z"}]

    monkeypatch.setattr(extractor, "_get", fake_get)
    assert len(list(extractor._iter_issues("moby/moby"))) == 101
    assert pages == [1, 2]


def test_stackoverflow_query_and_attribution_include_display_name(monkeypatch) -> None:
    assert "stackoverflow.users" in QUERY
    assert "u.display_name" in QUERY
    assert "SPLIT(LOWER(q.tags), '|')" in QUERY
    assert "CONCAT('%<', tag, '>%')" not in QUERY
    extractor = StackOverflowBigQueryExtractor(
        source_name="stackoverflow_docker",
        source_config={"license_url": "https://stackoverflow.com/help/licensing"},
        root_config={},
    )
    monkeypatch.setattr(extractor, "_rows", lambda: iter([{
        "question_id": 1,
        "answer_id": 2,
        "title": "Docker error",
        "question_body": "<p>Why?</p>",
        "answer_body": "<p>Fix it</p>",
        "question_score": 4,
        "answer_score": 5,
        "owner_user_id": 7,
        "display_name": "Ada",
        "tags": "<docker>",
    }]))
    record = next(extractor._records())
    assert record.author == "Ada"
    assert record.metadata["display_name"] == "Ada"
    assert "author Ada" in record.attribution


def test_stackoverflow_source_has_a_hard_query_cap() -> None:
    from docker_k8s_finetune.config import load_yaml

    source = load_yaml("config/sources.yaml")["sources"]["stackoverflow_docker"]
    assert source["maximum_bytes_billed_gib"] == 70
