from pathlib import Path

from docker_k8s_finetune.completeness import assess_pipeline_completeness


def test_completeness_reports_missing_upstream_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "sources.yaml").write_text(
        """version: 1
defaults:
  output_template: data/interim/extracted/{source}.jsonl
  idempotency: {manifest_suffix: .manifest.json}
sources:
  docs: {kind: git_markdown, tier: 1}
""",
        encoding="utf-8",
    )
    (config / "normalization.yaml").write_text(
        """reverse_instruction:
  review_config: config/reverse_instruction_review.yaml
  generated_sample: data/sample.jsonl
  generated_full: data/full.jsonl
  base_url_env: SYNTHETIC_LLM_BASE_URL
  model_env: SYNTHETIC_LLM_MODEL
""",
        encoding="utf-8",
    )
    (config / "reverse_instruction_review.yaml").write_text("status: pending\n", encoding="utf-8")
    (config / "dedupe.yaml").write_text(
        """inputs: [data/direct.jsonl, data/full.jsonl]
exact: {output: data/exact.jsonl, report: reports/exact.json}
approximate: {output: data/approx.jsonl, embeddings: data/embeddings.npz, report: reports/approx.json}
""",
        encoding="utf-8",
    )
    (config / "validation.yaml").write_text(
        """input: data/approx.jsonl
outputs: {accepted: data/validated.jsonl, report: reports/validation.json}
""",
        encoding="utf-8",
    )

    result = assess_pipeline_completeness(root=tmp_path, verify_hashes=False)

    codes = {item["code"] for item in result["issues"]}
    assert result["ready_for_final_split"] is False
    assert "source_output_missing" in codes
    assert "reverse_review_pending" in codes
    assert "reverse_full_missing" in codes
    assert "approximate_dedupe_report_missing" in codes

