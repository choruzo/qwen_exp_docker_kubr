from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import (
    load_all,
    load_yaml,
    validate_auxiliary_configs,
    validate_cross_config,
    validate_sources_config,
    validate_splits_config,
)
from .errors import PipelineError


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def command_config_validate(args: argparse.Namespace) -> int:
    configs = load_all(args.config_dir)
    auxiliary = validate_auxiliary_configs(args.config_dir)
    validate_cross_config(configs.sources, auxiliary)
    print(json.dumps({
        "sources": len(configs.sources["sources"]),
        "mass_extraction_allowed": configs.sources["mass_extraction_allowed"],
        "split_ratios": configs.splits["ratios"],
        "model": configs.training["model"]["name"],
        "auxiliary_configs": sorted(auxiliary),
        "benchmark_variants": sorted(auxiliary["benchmark"]["variants"]),
    }, indent=2))
    return 0


def command_pipeline_status(args: argparse.Namespace) -> int:
    from .completeness import assess_pipeline_completeness

    result = assess_pipeline_completeness(verify_hashes=not args.no_verify_hashes)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ready_for_final_split"] else 1


def command_extract(args: argparse.Namespace) -> int:
    from .extract.runner import run_extraction

    summary = run_extraction(
        config_path=Path(args.config),
        selected_sources=args.source,
        force=args.force,
        include_quarantine=args.include_quarantine,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def command_clean(args: argparse.Namespace) -> int:
    from .clean.pipeline import run_cleaning

    report = run_cleaning(
        config_path=Path(args.config),
        input_dir=Path(args.input_dir),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_normalize(args: argparse.Namespace) -> int:
    from .normalize.pipeline import prepare_reverse_candidates, run_direct_normalization

    result = {}
    if args.mode in ("all", "direct"):
        result["direct"] = run_direct_normalization(config_path=Path(args.config))
    if args.mode in ("all", "prepare-sample"):
        result["reverse_candidates"] = prepare_reverse_candidates(config_path=Path(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_normalize_generate(args: argparse.Namespace) -> int:
    from .normalize.generate import run_reverse_generation

    result = run_reverse_generation(scope=args.scope, config_path=Path(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_dedupe(args: argparse.Namespace) -> int:
    from .dedupe.approximate import run_approximate_dedupe
    from .dedupe.exact import run_exact_dedupe

    handler = run_exact_dedupe if args.mode == "exact" else run_approximate_dedupe
    result = handler(config_path=Path(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_validate_dataset(args: argparse.Namespace) -> int:
    from .validate.pipeline import run_syntax_validation

    result = run_syntax_validation(
        config_path=Path(args.config),
        input_override=Path(args.input) if args.input else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_split(args: argparse.Namespace) -> int:
    from .split.pipeline import run_split

    result = run_split(
        config_path=Path(args.config),
        skip_semantic_audit=args.skip_semantic_audit,
        allow_provisional=args.allow_provisional,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from .train.runner import run_training

    result = run_training(
        config_path=Path(args.config),
        smoke_test=args.smoke_test,
        export=not args.no_export,
        preflight_only=args.preflight,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    from .benchmark.pipeline import run_benchmark

    result = run_benchmark(
        variant=args.variant,
        config_path=Path(args.config),
        limit=args.limit,
        skip_judge=args.skip_judge,
        skip_syntax=args.skip_syntax,
    )
    print(json.dumps({
        "variant": result["variant"],
        "provisional": result["provisional"],
        "metrics": result["metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


def command_benchmark_judge(args: argparse.Namespace) -> int:
    from .benchmark.pipeline import run_benchmark_judge

    result = run_benchmark_judge(
        variant=args.variant,
        config_path=Path(args.config),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_benchmark_syntax(args: argparse.Namespace) -> int:
    from .benchmark.pipeline import run_benchmark_syntax

    result = run_benchmark_syntax(
        variant=args.variant,
        config_path=Path(args.config),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_report(args: argparse.Namespace) -> int:
    from .benchmark.report import run_report

    result = run_report(
        config_path=Path(args.config),
        allow_provisional=args.allow_provisional,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dkft", description="Docker/Kubernetes fine-tuning pipeline")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("pipeline-status", help="Audit end-to-end lineage and final-split readiness")
    status_parser.add_argument("--no-verify-hashes", action="store_true")
    status_parser.set_defaults(handler=command_pipeline_status)

    config_parser = subparsers.add_parser("config-validate", help="Validate all YAML configuration files")
    config_parser.add_argument("--config-dir", default="config")
    config_parser.set_defaults(handler=command_config_validate)

    extract_parser = subparsers.add_parser("extract", help="Extract configured sources to common JSONL")
    extract_parser.add_argument("--config", default="config/sources.yaml")
    extract_parser.add_argument("--source", action="append", help="Source id; repeat to select multiple")
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--include-quarantine", action="store_true")
    extract_parser.add_argument("--dry-run", action="store_true")
    extract_parser.set_defaults(handler=command_extract)

    clean_parser = subparsers.add_parser("clean", help="Clean extracted JSONL and redact PII")
    clean_parser.add_argument("--config", default="config/cleaning.yaml")
    clean_parser.add_argument("--input-dir", default="data/interim/extracted")
    clean_parser.set_defaults(handler=command_clean)

    normalize_parser = subparsers.add_parser("normalize", help="Normalize direct pairs and prepare reverse-instruction review")
    normalize_parser.add_argument("--config", default="config/normalization.yaml")
    normalize_parser.add_argument("--mode", choices=("all", "direct", "prepare-sample"), default="all")
    normalize_parser.set_defaults(handler=command_normalize)

    generate_parser = subparsers.add_parser("normalize-generate", help="Generate reverse-instruction sample or approved full set")
    generate_parser.add_argument("--config", default="config/normalization.yaml")
    generate_parser.add_argument("--scope", choices=("sample", "full"), default="sample")
    generate_parser.set_defaults(handler=command_normalize_generate)

    dedupe_parser = subparsers.add_parser("dedupe", help="Deduplicate normalized ChatML records")
    dedupe_parser.add_argument("--config", default="config/dedupe.yaml")
    dedupe_parser.add_argument("--mode", choices=("exact", "approximate"), default="exact")
    dedupe_parser.set_defaults(handler=command_dedupe)

    dataset_validation_parser = subparsers.add_parser("validate-dataset", help="Validate YAML and Dockerfiles in ChatML")
    dataset_validation_parser.add_argument("--config", default="config/validation.yaml")
    dataset_validation_parser.add_argument("--input", help="Provisional input override")
    dataset_validation_parser.set_defaults(handler=command_validate_dataset)

    split_parser = subparsers.add_parser("split", help="Create stratified train/validation/test splits")
    split_parser.add_argument("--config", default="config/splits.yaml")
    split_parser.add_argument("--skip-semantic-audit", action="store_true", help="Provisional use only")
    split_parser.add_argument("--allow-provisional", action="store_true", help="Allow incomplete upstream lineage")
    split_parser.set_defaults(handler=command_split)

    train_parser = subparsers.add_parser("train", help="Run text-only QLoRA with Qwen3.5-4B")
    train_parser.add_argument("--config", default="config/training.yaml")
    train_parser.add_argument("--smoke-test", action="store_true")
    train_parser.add_argument("--no-export", action="store_true")
    train_parser.add_argument("--preflight", action="store_true")
    train_parser.set_defaults(handler=command_train)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the frozen before/after benchmark")
    benchmark_parser.add_argument(
        "--variant",
        choices=("baseline", "finetuned_safetensors", "finetuned_gguf"),
        required=True,
    )
    benchmark_parser.add_argument("--config", default="config/benchmark.yaml")
    benchmark_parser.add_argument("--limit", type=int, help="Provisional smoke limit")
    benchmark_parser.add_argument("--skip-judge", action="store_true", help="Provisional use only")
    benchmark_parser.add_argument("--skip-syntax", action="store_true", help="Provisional use only")
    benchmark_parser.set_defaults(handler=command_benchmark)

    syntax_parser = subparsers.add_parser(
        "benchmark-syntax", help="Validate saved predictions with host Docker validators"
    )
    syntax_parser.add_argument(
        "--variant",
        choices=("baseline", "finetuned_safetensors", "finetuned_gguf"),
        required=True,
    )
    syntax_parser.add_argument("--config", default="config/benchmark.yaml")
    syntax_parser.set_defaults(handler=command_benchmark_syntax)

    judge_parser = subparsers.add_parser(
        "benchmark-judge", help="Judge saved full predictions after releasing the evaluated model"
    )
    judge_parser.add_argument(
        "--variant",
        choices=("baseline", "finetuned_safetensors", "finetuned_gguf"),
        required=True,
    )
    judge_parser.add_argument("--config", default="config/benchmark.yaml")
    judge_parser.set_defaults(handler=command_benchmark_judge)

    report_parser = subparsers.add_parser("report", help="Build the before/after comparison report")
    report_parser.add_argument("--config", default="config/benchmark.yaml")
    report_parser.add_argument("--allow-provisional", action="store_true")
    report_parser.set_defaults(handler=command_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.handler(args))
    except (PipelineError, OSError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
