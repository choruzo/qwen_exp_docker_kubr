from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..config import REQUIRED_CATEGORIES, load_yaml
from ..errors import PipelineError
from ..io import read_jsonl
from ..train.core import verify_export_artifact
from .core import aggregate_results, judge_provenance, syntax_provenance


def _validate_frozen_record_coverage(
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    root: Path,
) -> None:
    inputs = config.get("inputs", {})
    if not all(inputs.get(key) for key in ("test_manifest", "out_of_domain")):
        return
    test_manifest = list(read_jsonl(root / str(inputs["test_manifest"])))
    ood = list(read_jsonl(root / str(inputs["out_of_domain"])))
    expected_hashes = [str(item["content_hash"]) for item in test_manifest] + [
        str(item.get("meta", {}).get("content_hash", "")) for item in ood
    ]
    if any(not value for value in expected_hashes) or len(set(expected_hashes)) != len(expected_hashes):
        raise PipelineError("Current frozen benchmark inputs contain invalid or duplicate hashes")
    for variant, result in results.items():
        records = result.get("records")
        if not isinstance(records, list):
            raise PipelineError(f"Benchmark result lacks record-level evidence: {variant}")
        actual_hashes = [str(record.get("content_hash", "")) for record in records]
        if actual_hashes != expected_hashes:
            raise PipelineError(
                f"Benchmark records do not exactly match frozen test plus OOD order: {variant}"
            )
        if result.get("metrics") != aggregate_results(records):
            raise PipelineError(f"Benchmark metrics do not match record-level evidence: {variant}")


def _validate_export_provenance(
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    root: Path,
) -> None:
    variants = config.get("variants", {})
    for variant in ("finetuned_safetensors", "finetuned_gguf"):
        variant_config = variants.get(variant, {})
        manifest = variant_config.get("export_manifest")
        artifact = variant_config.get("artifact")
        if manifest is None or artifact is None:
            continue
        expected = verify_export_artifact(root, Path(str(manifest)), str(artifact))
        stored = results[variant].get("model_provenance")
        if not isinstance(stored, Mapping):
            raise PipelineError(f"Benchmark lacks model provenance: {variant}")
        artifact_provenance = dict(stored)
        served_model = artifact_provenance.pop("served_model", None)
        if artifact_provenance != expected:
            raise PipelineError(
                f"Benchmark model provenance differs from current export artifact: {variant}"
            )
        if variant == "finetuned_gguf" and variant_config.get("require_llama_props"):
            runtime = results[variant].get("runtime")
            runtime_served = runtime.get("served_model") if isinstance(runtime, Mapping) else None
            if not isinstance(served_model, Mapping) or served_model != runtime_served:
                raise PipelineError("GGUF benchmark lacks consistent served-model identity")
            matches = [
                entry for entry in expected["files"]
                if entry.get("path") == served_model.get("model_path")
                and entry.get("sha256") == served_model.get("sha256")
            ]
            if len(matches) != 1 or served_model.get("quantization") != variant_config.get(
                "expected_quantization"
            ):
                raise PipelineError("GGUF benchmark did not serve the configured manifested quantization")


def _validate_training_export_lineage(
    training_summary: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    root: Path,
) -> None:
    variant = config.get("variants", {}).get("finetuned_safetensors", {})
    configured_manifest = variant.get("export_manifest")
    if configured_manifest is None:
        return
    manifest_path = root / str(configured_manifest)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read export lineage manifest: {exc}") from exc
    training_split = training_summary.get("training_split")
    if not isinstance(training_split, Mapping) or manifest.get("training_split") != training_split:
        raise PipelineError("Training result and export manifest use different data splits")
    test_hashes = {result.get("split_test_sha256") for result in results.values()}
    if len(test_hashes) != 1 or training_split.get("test") not in test_hashes:
        raise PipelineError("Training/export lineage differs from benchmark test split")
    if manifest.get("training_length_filter") != training_summary.get("length_filter"):
        raise PipelineError("Training result and export manifest use different length filters")
    if manifest.get("base_model") != training_summary.get("base_model"):
        raise PipelineError("Training result and export manifest use different base-model provenance")
    expected_contract = {
        "fingerprint": training_summary.get("checkpoint_fingerprint"),
        "contract": training_summary.get("checkpoint_contract"),
    }
    if (
        not expected_contract["fingerprint"]
        or not isinstance(expected_contract["contract"], Mapping)
        or manifest.get("training_contract") != expected_contract
    ):
        raise PipelineError("Training result and export manifest use different training contracts")


def _mean(result: Mapping[str, Any], category: str, metric: str) -> float | None:
    value = result["metrics"]["by_category"].get(category, {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


def _overall_mean(result: Mapping[str, Any], metric: str) -> float | None:
    value = result.get("metrics", {}).get("overall", {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


def _judge_ci(result: Mapping[str, Any], category: str) -> str:
    value = result.get("metrics", {}).get("by_category", {}).get(category, {}).get("llm_judge", {})
    mean, low, high = value.get("mean"), value.get("ci95_low"), value.get("ci95_high")
    if mean is None or low is None or high is None:
        return "n/a"
    return f"{float(mean):.3f} [{float(low):.3f}, {float(high):.3f}]"


def _fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%" if percent else f"{value:.3f}"


def _change(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    if before == 0:
        return f"{(after - before) * 100:+.2f} pp"
    return f"{((after / before) - 1) * 100:+.2f}%"


def _truncation_gate(result: Mapping[str, Any]) -> tuple[float | None, str]:
    audit = result.get("truncation_audit")
    if not isinstance(audit, Mapping):
        return None, "n/a"
    threshold = audit.get("max_allowed_rate")
    limit = float(threshold) if isinstance(threshold, (int, float)) else None
    return limit, "PASS" if audit.get("passed") is True else "FAIL"


def _runtime_label(result: Mapping[str, Any]) -> str:
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        return "n/a"
    device = runtime.get("device")
    if device:
        capability = runtime.get("compute_capability")
        suffix = f" (CUDA sm_{str(capability).replace('.', '')})" if capability else ""
        suffix = f" (ROCm {runtime['gcn_arch']})" if runtime.get("gcn_arch") else suffix
        return f"{device}{suffix}"
    served = runtime.get("served_model")
    if isinstance(served, Mapping) and served.get("alias"):
        return f"llama.cpp: {served['alias']} ({served.get('quantization', 'unknown')})"
    if runtime.get("model"):
        return f"{runtime.get('backend', 'endpoint')}: {runtime['model']}"
    return str(runtime.get("backend", "n/a"))


def _in_process_runtime_signature(result: Mapping[str, Any], variant: str) -> tuple[str, ...]:
    """Return a strict, accelerator-aware signature for direct model inference."""
    runtime = result.get("runtime")
    common_fields = ("backend", "device", "torch_version")
    if not isinstance(runtime, Mapping) or any(
        not str(runtime.get(field, "")).strip() for field in common_fields
    ):
        raise PipelineError(f"Benchmark result lacks complete runtime identity: {variant}")
    accelerator = str(runtime.get("accelerator") or (
        "cuda" if runtime.get("compute_capability") and runtime.get("torch_cuda_version") else ""
    ))
    accelerator_fields = {
        "cuda": ("compute_capability", "torch_cuda_version"),
        "rocm": ("gcn_arch", "torch_hip_version"),
    }.get(accelerator)
    if accelerator_fields is None or any(
        not str(runtime.get(field, "")).strip() for field in accelerator_fields
    ):
        raise PipelineError(f"Benchmark result lacks complete {accelerator} runtime identity: {variant}")
    return (*tuple(str(runtime[field]) for field in common_fields), accelerator,
            *tuple(str(runtime[field]) for field in accelerator_fields))


def build_markdown(
    baseline: Mapping[str, Any],
    finetuned: Mapping[str, Any],
    gguf: Mapping[str, Any],
    *,
    loss_chart_path: str | None,
    training_summary: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "# Comparacion Qwen3.5-4B Docker/Kubernetes",
        "",
        f"- Baseline: {baseline.get('created_at', 'unknown')}",
        f"- Fine-tuned safetensors: {finetuned.get('created_at', 'unknown')}",
        f"- Fine-tuned GGUF: {gguf.get('created_at', 'unknown')}",
        "",
        "## Resultados por categoria",
        "",
        "| Categoria | Exact base | Exact FT | Semantica base | Semantica FT | Cambio sem. | Juez base | Juez FT | Sintaxis base | Sintaxis FT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    regressions: list[str] = []
    for category in sorted(REQUIRED_CATEGORIES):
        base_sem = _mean(baseline, category, "semantic_similarity")
        fine_sem = _mean(finetuned, category, "semantic_similarity")
        base_exact = _mean(baseline, category, "exact_match")
        fine_exact = _mean(finetuned, category, "exact_match")
        base_judge = _mean(baseline, category, "llm_judge")
        fine_judge = _mean(finetuned, category, "llm_judge")
        base_syntax = _mean(baseline, category, "syntax_validity")
        fine_syntax = _mean(finetuned, category, "syntax_validity")
        lines.append(
            f"| {category} | {_fmt(base_exact, percent=True)} | {_fmt(fine_exact, percent=True)} "
            f"| {_fmt(base_sem)} | {_fmt(fine_sem)} | {_change(base_sem, fine_sem)} "
            f"| {_fmt(base_judge)} | {_fmt(fine_judge)} | {_fmt(base_syntax, percent=True)} "
            f"| {_fmt(fine_syntax, percent=True)} |"
        )
        if base_exact is not None and fine_exact is not None and fine_exact < base_exact:
            regressions.append(f"{category}: exact match {base_exact:.3f} -> {fine_exact:.3f}")
        if base_sem is not None and fine_sem is not None and fine_sem < base_sem:
            regressions.append(f"{category}: similitud semantica {base_sem:.3f} -> {fine_sem:.3f}")
        if base_judge is not None and fine_judge is not None and fine_judge < base_judge:
            regressions.append(f"{category}: juez {base_judge:.3f} -> {fine_judge:.3f}")
        if base_syntax is not None and fine_syntax is not None and fine_syntax < base_syntax:
            regressions.append(f"{category}: validez sintactica {base_syntax:.3f} -> {fine_syntax:.3f}")

    lines.extend([
        "",
        "## Efecto de la cuantizacion GGUF",
        "",
        "| Categoria | Exact safetensors | Exact GGUF | Semantica safetensors | Semantica GGUF | Cambio sem. | Juez safetensors | Juez GGUF | Sintaxis safetensors | Sintaxis GGUF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for category in sorted(REQUIRED_CATEGORIES | {"out_of_domain"}):
        fine_sem = _mean(finetuned, category, "semantic_similarity")
        gguf_sem = _mean(gguf, category, "semantic_similarity")
        fine_exact = _mean(finetuned, category, "exact_match")
        gguf_exact = _mean(gguf, category, "exact_match")
        fine_judge = _mean(finetuned, category, "llm_judge")
        gguf_judge = _mean(gguf, category, "llm_judge")
        fine_syntax = _mean(finetuned, category, "syntax_validity")
        gguf_syntax = _mean(gguf, category, "syntax_validity")
        lines.append(
            f"| {category} | {_fmt(fine_exact, percent=True)} | {_fmt(gguf_exact, percent=True)} "
            f"| {_fmt(fine_sem)} | {_fmt(gguf_sem)} | {_change(fine_sem, gguf_sem)} "
            f"| {_fmt(fine_judge)} | {_fmt(gguf_judge)} | {_fmt(fine_syntax, percent=True)} "
            f"| {_fmt(gguf_syntax, percent=True)} |"
        )
        for label, before, after in (
            ("exact match", fine_exact, gguf_exact),
            ("similitud semantica", fine_sem, gguf_sem),
            ("juez", fine_judge, gguf_judge),
            ("validez sintactica", fine_syntax, gguf_syntax),
        ):
            if before is not None and after is not None and after < before:
                regressions.append(
                    f"cuantizacion {category}: {label} {before:.3f} -> {after:.3f}"
                )

    base_yaml = _mean(baseline, "generacion_yaml", "syntax_validity")
    fine_yaml = _mean(finetuned, "generacion_yaml", "syntax_validity")
    base_docker = _mean(baseline, "dockerfile", "syntax_validity")
    fine_docker = _mean(finetuned, "dockerfile", "syntax_validity")
    base_ood = _mean(baseline, "out_of_domain", "llm_judge")
    fine_ood = _mean(finetuned, "out_of_domain", "llm_judge")
    gguf_ood = _mean(gguf, "out_of_domain", "llm_judge")
    base_ood_sem = _mean(baseline, "out_of_domain", "semantic_similarity")
    fine_ood_sem = _mean(finetuned, "out_of_domain", "semantic_similarity")
    gguf_ood_sem = _mean(gguf, "out_of_domain", "semantic_similarity")
    base_ood_exact = _mean(baseline, "out_of_domain", "exact_match")
    fine_ood_exact = _mean(finetuned, "out_of_domain", "exact_match")
    gguf_ood_exact = _mean(gguf, "out_of_domain", "exact_match")
    base_cli_exact = _mean(baseline, "comando_cli", "exact_match")
    fine_cli_exact = _mean(finetuned, "comando_cli", "exact_match")
    for label, before, after in (
        ("exact match", base_ood_exact, fine_ood_exact),
        ("similitud semantica", base_ood_sem, fine_ood_sem),
        ("juez", base_ood, fine_ood),
    ):
        if before is not None and after is not None and after < before:
            regressions.append(
                f"fuera de dominio: {label} {before:.3f} -> {after:.3f}"
            )
    lines.extend([
        "",
        "## Indicadores clave",
        "",
        f"- Validez YAML: {_fmt(base_yaml, percent=True)} -> {_fmt(fine_yaml, percent=True)} ({_change(base_yaml, fine_yaml)}).",
        f"- Validez Dockerfile: {_fmt(base_docker, percent=True)} -> {_fmt(fine_docker, percent=True)} ({_change(base_docker, fine_docker)}).",
        f"- Exact match de comandos CLI: {_fmt(base_cli_exact, percent=True)} -> {_fmt(fine_cli_exact, percent=True)} ({_change(base_cli_exact, fine_cli_exact)}).",
        f"- Fuera de dominio, LLM-juez: {_fmt(base_ood)} -> {_fmt(fine_ood)} ({_change(base_ood, fine_ood)}).",
        "",
        "## Conservacion fuera de dominio",
        "",
        "| Variante | Exact match | Similitud semantica | LLM-juez |",
        "|---|---:|---:|---:|",
        f"| Baseline | {_fmt(base_ood_exact, percent=True)} | {_fmt(base_ood_sem)} | {_fmt(base_ood)} |",
        f"| Fine-tuned safetensors | {_fmt(fine_ood_exact, percent=True)} | {_fmt(fine_ood_sem)} | {_fmt(fine_ood)} |",
        f"| Fine-tuned GGUF | {_fmt(gguf_ood_exact, percent=True)} | {_fmt(gguf_ood_sem)} | {_fmt(gguf_ood)} |",
        "",
        "## Incertidumbre del LLM-juez (media e IC95)",
        "",
        "| Categoria | Baseline | Fine-tuned | GGUF |",
        "|---|---:|---:|---:|",
    ])
    for category in sorted(REQUIRED_CATEGORIES | {"out_of_domain"}):
        lines.append(
            f"| {category} | {_judge_ci(baseline, category)} | {_judge_ci(finetuned, category)} "
            f"| {_judge_ci(gguf, category)} |"
        )
    lines.extend([
        "",
        "## Rendimiento en el hardware evaluado",
        "",
        "La latencia y los tokens/s por solicitud reflejan el tiempo compartido que observa cada elemento del lote. "
        "El throughput de lote suma los tokens de todas las solicitudes concurrentes y es la medida de capacidad del hardware.",
        "",
        "| Variante | Runtime/dispositivo | Lote configurado/real medio | Latencia/solicitud (s) | Tokens/s por solicitud | Throughput lote (tokens/s) |",
        "|---|---|---:|---:|---:|---:|",
        f"| Baseline | {_runtime_label(baseline)} | {baseline.get('generation', {}).get('batch_size', 1)} / {_fmt(_overall_mean(baseline, 'generation_batch_size'))} | {_fmt(_overall_mean(baseline, 'latency_seconds'))} | {_fmt(_overall_mean(baseline, 'tokens_per_second'))} | {_fmt(_overall_mean(baseline, 'batch_tokens_per_second'))} |",
        f"| Fine-tuned safetensors | {_runtime_label(finetuned)} | {finetuned.get('generation', {}).get('batch_size', 1)} / {_fmt(_overall_mean(finetuned, 'generation_batch_size'))} | {_fmt(_overall_mean(finetuned, 'latency_seconds'))} | {_fmt(_overall_mean(finetuned, 'tokens_per_second'))} | {_fmt(_overall_mean(finetuned, 'batch_tokens_per_second'))} |",
        f"| Fine-tuned GGUF | {_runtime_label(gguf)} | {gguf.get('generation', {}).get('batch_size', 1)} / {_fmt(_overall_mean(gguf, 'generation_batch_size'))} | {_fmt(_overall_mean(gguf, 'latency_seconds'))} | {_fmt(_overall_mean(gguf, 'tokens_per_second'))} | {_fmt(_overall_mean(gguf, 'batch_tokens_per_second'))} |",
    ])
    lines.extend([
        "",
        "## Integridad de la generacion",
        "",
        "| Variante | Respuestas con reintento | Respuestas truncadas | Umbral maximo | Gate |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, result in (
        ("Baseline", baseline),
        ("Fine-tuned safetensors", finetuned),
        ("Fine-tuned GGUF", gguf),
    ):
        threshold, gate = _truncation_gate(result)
        lines.append(
            f"| {label} | {_fmt(_overall_mean(result, 'retry_rate'), percent=True)} "
            f"| {_fmt(_overall_mean(result, 'truncation_rate'), percent=True)} "
            f"| {_fmt(threshold, percent=True)} | {gate} |"
        )
    if training_summary:
        vram = training_summary.get("vram") or {}
        length_filter = training_summary.get("length_filter") or {}
        train_filter = length_filter.get("train") or {}
        validation_filter = length_filter.get("validation") or {}
        lines.extend([
            "",
            "## Entrenamiento ejecutado",
            "",
            f"- Contexto maximo: {training_summary.get('max_seq_length', 'n/a')} tokens.",
            f"- Batch por dispositivo: {training_summary.get('batch_size', 'n/a')}; batch efectivo: {training_summary.get('effective_batch_size', 'n/a')}.",
            f"- Train loss final: {_fmt(training_summary.get('metrics', {}).get('train_loss'))}.",
            f"- Validation loss final: {_fmt(training_summary.get('evaluation_metrics', {}).get('eval_loss'))}.",
            f"- VRAM pico asignada/reservada: {_fmt(vram.get('peak_allocated_gib'))} / {_fmt(vram.get('peak_reserved_gib'))} GiB.",
            f"- Ejemplos excluidos por longitud: train {train_filter.get('excluded_records', 'n/a')} de {train_filter.get('input_records', 'n/a')}; validacion {validation_filter.get('excluded_records', 'n/a')} de {validation_filter.get('input_records', 'n/a')}.",
            f"- Huella del subconjunto train: `{train_filter.get('kept_content_hashes_sha256', 'n/a')}`.",
            f"- Huella del subconjunto validacion: `{validation_filter.get('kept_content_hashes_sha256', 'n/a')}`.",
        ])
    if loss_chart_path:
        lines.extend(["", "## Curvas de entrenamiento", "", f"![Loss train/validation]({loss_chart_path})"])
    lines.extend(["", "## Conclusion honesta", ""])
    if regressions:
        lines.append("Se observaron regresiones y no deben ocultarse:")
        lines.extend(f"- {value}" for value in regressions)
        lines.append("")
        lines.append(
            "Hipotesis a comprobar: cobertura insuficiente de la categoria, ejemplos sinteticos de baja calidad, "
            "sobreajuste o degradacion por cuantizacion. Debe ajustarse el muestreo o los hiperparametros y repetir "
            "exactamente el mismo benchmark congelado."
        )
    else:
        lines.append(
            "No se observaron regresiones en las metricas disponibles. Esta conclusion solo es valida para el test "
            "congelado, la rubrica y las versiones de herramientas registradas en los JSON de resultados."
        )
    lines.append("")
    return "\n".join(lines)


def render_loss_svg(log_history: list[Mapping[str, Any]], output: Path) -> bool:
    train = [(float(item["step"]), float(item["loss"])) for item in log_history if "step" in item and "loss" in item]
    validation = [(float(item["step"]), float(item["eval_loss"])) for item in log_history if "step" in item and "eval_loss" in item]
    if not train or not validation:
        return False
    points = train + validation
    width, height, pad = 900, 420, 55
    max_x = max(value[0] for value in points) or 1.0
    values = [value[1] for value in points]
    min_y, max_y = min(values), max(values)
    if min_y == max_y:
        max_y = min_y + 1.0

    def coordinates(series: list[tuple[float, float]]) -> str:
        return " ".join(
            f"{pad + x / max_x * (width - 2 * pad):.2f},{height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad):.2f}"
            for x, y in series
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>'
        f'<polyline points="{coordinates(train)}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<polyline points="{coordinates(validation)}" fill="none" stroke="#dc2626" stroke-width="2"/>'
        '<text x="70" y="28" font-family="sans-serif" font-size="14" fill="#2563eb">train loss</text>'
        '<text x="180" y="28" font-family="sans-serif" font-size="14" fill="#dc2626">validation loss</text>'
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">step</text>'
        '</svg>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg + "\n", encoding="utf-8")
    return True


def run_report(
    *, config_path: Path = Path("config/benchmark.yaml"), root: Path = Path("."),
    allow_provisional: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    results = {}
    for variant in ("baseline", "finetuned_safetensors", "finetuned_gguf"):
        path = root / str(config["outputs"][variant])
        if not path.is_file():
            raise PipelineError(f"Benchmark result is missing: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("variant") != variant:
            raise PipelineError(f"Benchmark result variant mismatch: {variant}")
        if result.get("provisional") and not allow_provisional:
            raise PipelineError(f"Refusing to report provisional benchmark: {variant}")
        completion = result.get("completion") or {}
        if not allow_provisional and not all(
            completion.get(stage) is True
            for stage in ("full_test", "generation", "syntax", "judge")
        ):
            raise PipelineError(f"Benchmark result has incomplete stages: {variant}")
        results[variant] = result
    _validate_frozen_record_coverage(results, config, root)
    _validate_export_provenance(results, config, root)
    test_hashes = {str(result.get("split_test_sha256") or "") for result in results.values()}
    if len(test_hashes) != 1 or "" in test_hashes:
        raise PipelineError("Benchmark variants were not run against the same frozen test split")
    runtime_signatures = {
        variant: _in_process_runtime_signature(results[variant], variant)
        for variant in ("baseline", "finetuned_safetensors")
    }
    if len(set(runtime_signatures.values())) != 1:
        raise PipelineError("Baseline and safetensors benchmarks use different accelerator runtimes")
    if config.get("generation") and any(
        result.get("generation") != config["generation"] for result in results.values()
    ):
        raise PipelineError("Benchmark variants use different generation parameters")
    if config.get("semantic_similarity") and any(
        result.get("semantic_similarity_config") != config["semantic_similarity"]
        for result in results.values()
    ):
        raise PipelineError("Benchmark variants use different semantic similarity models")
    if config.get("syntax_validation", {}).get("enabled"):
        expected_syntax_provenance = syntax_provenance(config, root)
        if any(
            result.get("syntax_provenance") != expected_syntax_provenance
            for result in results.values()
        ):
            raise PipelineError("Benchmark variants use different syntax validator configurations")
    judge_config = config.get("llm_judge", {})
    if judge_config.get("enabled"):
        expected_provenance = judge_provenance(config, root)
        if any(
            result.get("judge_provenance") != expected_provenance
            for result in results.values()
        ):
            raise PipelineError("Benchmark variants use different judge prompt or manifest")
        expected_identity = judge_config.get("model_identity")
        if expected_identity:
            for variant, result in results.items():
                runtime = result.get("judge_runtime")
                served = runtime.get("served") if isinstance(runtime, Mapping) else None
                if (
                    not isinstance(runtime, Mapping)
                    or runtime.get("expected") != expected_identity
                    or not isinstance(served, Mapping)
                    or any(
                        served.get(field) != expected_identity.get(field)
                        for field in ("alias", "filename", "quantization")
                    )
                ):
                    raise PipelineError(
                        f"Benchmark result lacks fixed judge model identity: {variant}"
                    )

    chart = root / str(config["outputs"]["loss_chart"])
    chart_created = False
    training_summary: Mapping[str, Any] | None = None
    training_result = root / str(
        config.get("inputs", {}).get(
            "training_result", "artifacts/metrics/training_result.json"
        )
    )
    if training_result.is_file():
        state = json.loads(training_result.read_text(encoding="utf-8"))
        training_summary = state
        chart_created = render_loss_svg(state.get("log_history", []), chart)
    elif not allow_provisional:
        raise PipelineError(f"Final report requires training result: {training_result}")
    if not chart_created:
        checkpoints = training_result.parent.parent / "checkpoints"
        trainer_states = sorted(
            checkpoints.glob("**/trainer_state.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if trainer_states:
            state = json.loads(trainer_states[-1].read_text(encoding="utf-8"))
            chart_created = render_loss_svg(state.get("log_history", []), chart)
    if not chart_created and not allow_provisional:
        raise PipelineError("Final report requires persisted train and validation loss history")
    if not allow_provisional and training_summary is not None:
        if not isinstance(training_summary.get("vram"), Mapping):
            raise PipelineError("Final report requires persisted VRAM metrics")
        length_filter = training_summary.get("length_filter")
        if not isinstance(length_filter, Mapping) or not all(
            isinstance(length_filter.get(split), Mapping) for split in ("train", "validation")
        ):
            raise PipelineError("Final report requires token-length filter provenance")
        _validate_training_export_lineage(training_summary, results, config, root)
    report_path = root / str(config["outputs"]["comparison_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        results["baseline"],
        results["finetuned_safetensors"],
        results["finetuned_gguf"],
        loss_chart_path=chart.name if chart_created else None,
        training_summary=training_summary,
    )
    report_path.write_text(markdown, encoding="utf-8", newline="\n")
    return {"report": str(report_path), "loss_chart": str(chart) if chart_created else None}
