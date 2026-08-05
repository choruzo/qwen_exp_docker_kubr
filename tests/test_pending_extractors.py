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
    result = extractor._resolution({"comments_url": "https://api.github.test/comments"})
    assert calls == [1, 2]
    assert result["body"] == "fixed"


def test_stackoverflow_query_and_attribution_include_display_name(monkeypatch) -> None:
    assert "stackoverflow.users" in QUERY
    assert "u.display_name" in QUERY
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

