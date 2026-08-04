from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..config import REQUIRED_CATEGORIES, load_yaml
from ..errors import PipelineError


def _mean(result: Mapping[str, Any], category: str, metric: str) -> float | None:
    value = result["metrics"]["by_category"].get(category, {}).get(metric, {}).get("mean")
    return float(value) if value is not None else None


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


def build_markdown(
    baseline: Mapping[str, Any],
    finetuned: Mapping[str, Any],
    gguf: Mapping[str, Any],
    *,
    loss_chart_path: str | None,
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
        "| Categoria | Semantica base | Semantica FT | Cambio | Juez base | Juez FT | Sintaxis base | Sintaxis FT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    regressions: list[str] = []
    for category in sorted(REQUIRED_CATEGORIES):
        base_sem = _mean(baseline, category, "semantic_similarity")
        fine_sem = _mean(finetuned, category, "semantic_similarity")
        base_judge = _mean(baseline, category, "llm_judge")
        fine_judge = _mean(finetuned, category, "llm_judge")
        base_syntax = _mean(baseline, category, "syntax_validity")
        fine_syntax = _mean(finetuned, category, "syntax_validity")
        lines.append(
            f"| {category} | {_fmt(base_sem)} | {_fmt(fine_sem)} | {_change(base_sem, fine_sem)} "
            f"| {_fmt(base_judge)} | {_fmt(fine_judge)} | {_fmt(base_syntax, percent=True)} "
            f"| {_fmt(fine_syntax, percent=True)} |"
        )
        if base_sem is not None and fine_sem is not None and fine_sem < base_sem:
            regressions.append(f"{category}: similitud semantica {base_sem:.3f} -> {fine_sem:.3f}")
        if base_judge is not None and fine_judge is not None and fine_judge < base_judge:
            regressions.append(f"{category}: juez {base_judge:.3f} -> {fine_judge:.3f}")

    lines.extend([
        "",
        "## Efecto de la cuantizacion GGUF",
        "",
        "| Categoria | Semantica safetensors | Semantica GGUF | Cambio | Juez safetensors | Juez GGUF |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for category in sorted(REQUIRED_CATEGORIES):
        fine_sem = _mean(finetuned, category, "semantic_similarity")
        gguf_sem = _mean(gguf, category, "semantic_similarity")
        fine_judge = _mean(finetuned, category, "llm_judge")
        gguf_judge = _mean(gguf, category, "llm_judge")
        lines.append(
            f"| {category} | {_fmt(fine_sem)} | {_fmt(gguf_sem)} | {_change(fine_sem, gguf_sem)} "
            f"| {_fmt(fine_judge)} | {_fmt(gguf_judge)} |"
        )

    base_yaml = _mean(baseline, "generacion_yaml", "syntax_validity")
    fine_yaml = _mean(finetuned, "generacion_yaml", "syntax_validity")
    base_docker = _mean(baseline, "dockerfile", "syntax_validity")
    fine_docker = _mean(finetuned, "dockerfile", "syntax_validity")
    base_ood = _mean(baseline, "out_of_domain", "llm_judge")
    fine_ood = _mean(finetuned, "out_of_domain", "llm_judge")
    lines.extend([
        "",
        "## Indicadores clave",
        "",
        f"- Validez YAML: {_fmt(base_yaml, percent=True)} -> {_fmt(fine_yaml, percent=True)} ({_change(base_yaml, fine_yaml)}).",
        f"- Validez Dockerfile: {_fmt(base_docker, percent=True)} -> {_fmt(fine_docker, percent=True)} ({_change(base_docker, fine_docker)}).",
        f"- Fuera de dominio, LLM-juez: {_fmt(base_ood)} -> {_fmt(fine_ood)} ({_change(base_ood, fine_ood)}).",
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
    points = train + validation
    if not points:
        return False
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
        if result.get("provisional") and not allow_provisional:
            raise PipelineError(f"Refusing to report provisional benchmark: {variant}")
        results[variant] = result

    chart = root / str(config["outputs"]["loss_chart"])
    trainer_states = sorted(
        (root / "artifacts" / "checkpoints").glob("**/trainer_state.json"),
        key=lambda path: path.stat().st_mtime,
    )
    chart_created = False
    if trainer_states:
        state = json.loads(trainer_states[-1].read_text(encoding="utf-8"))
        chart_created = render_loss_svg(state.get("log_history", []), chart)
    report_path = root / str(config["outputs"]["comparison_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        results["baseline"],
        results["finetuned_safetensors"],
        results["finetuned_gguf"],
        loss_chart_path=chart.name if chart_created else None,
    )
    report_path.write_text(markdown, encoding="utf-8", newline="\n")
    return {"report": str(report_path), "loss_chart": str(chart) if chart_created else None}
