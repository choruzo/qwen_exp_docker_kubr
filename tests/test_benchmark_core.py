from __future__ import annotations

import pytest

from docker_k8s_finetune.benchmark.core import (
    aggregate_results,
    exact_match,
    parse_judge_payload,
)
from docker_k8s_finetune.benchmark.report import build_markdown
from docker_k8s_finetune.errors import PipelineError


def _result(value: float) -> dict:
    metric = {"count": 1, "mean": value, "stddev": 0.0, "ci95_low": value, "ci95_high": value}
    categories = {
        category: {
            "semantic_similarity": metric,
            "llm_judge": metric,
            "syntax_validity": metric,
        }
        for category in ("concepto", "comando_cli", "generacion_yaml", "troubleshooting", "dockerfile", "arquitectura", "out_of_domain")
    }
    return {"created_at": "2026-08-04T00:00:00Z", "metrics": {"by_category": categories}}


def test_exact_match_unwraps_shell_fence() -> None:
    assert exact_match("kubectl get pods", " " + chr(96) * 3 + "bash\nkubectl get pods\n" + chr(96) * 3) == 1.0


def test_judge_scores_are_strictly_bounded() -> None:
    assert parse_judge_payload({"correctness": 5, "completeness": 3})["mean"] == 4.0
    with pytest.raises(PipelineError, match=r"\[1, 5\]"):
        parse_judge_payload({"correctness": 6, "completeness": 3})


def test_aggregate_reports_category_metrics() -> None:
    record = {
        "category": "comando_cli", "error": None, "exact_match": 1.0,
        "semantic_similarity": 0.8, "syntax_valid": None,
        "judge": {"mean": 4.0}, "latency_seconds": 2.0, "tokens_per_second": 20.0,
    }
    summary = aggregate_results([record])
    assert summary["by_category"]["comando_cli"]["semantic_similarity"]["mean"] == 0.8


def test_report_calls_out_regressions() -> None:
    markdown = build_markdown(_result(0.8), _result(0.7), _result(0.69), loss_chart_path=None)
    assert "regresiones" in markdown
    assert "concepto" in markdown
    assert "Incertidumbre del LLM-juez" in markdown
    assert "Rendimiento en el hardware evaluado" in markdown
