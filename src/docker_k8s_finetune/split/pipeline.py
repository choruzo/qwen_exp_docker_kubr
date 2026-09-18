from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..completeness import assess_pipeline_completeness
from ..config import load_yaml, validate_splits_config
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, read_jsonl
from ..schema import utc_now_iso
from .audit import audit_and_repair
from .core import allocate_stratum, deterministic_key, hard_complexity, split_statistics


_LICENSE_URLS = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "BSD-2-Clause": "https://opensource.org/license/bsd-2-clause",
    "BSD-3-Clause": "https://opensource.org/license/bsd-3-clause",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "ISC": "https://opensource.org/license/isc-license-txt",
    "MIT": "https://opensource.org/license/mit",
    "Unlicense": "https://unlicense.org/",
    "Zlib": "https://opensource.org/license/zlib",
}


def _required_meta(record: dict[str, Any], field: str) -> Any:
    key = field.removeprefix("meta.")
    value = record.get("meta", {}).get(key)
    if value in (None, ""):
        raise PipelineError(f"Split input record is missing {field}")
    return value


def _rebalance_categories(
    *, split_name: str, assignments: dict[str, str], group_info: dict[str, dict[str, Any]],
    target_count: int, minimum_fractions: dict[str, float], seed: int,
) -> None:
    def counts() -> tuple[int, Counter[str]]:
        category_counts: Counter[str] = Counter()
        total = 0
        for group, assigned in assignments.items():
            if assigned != split_name:
                continue
            size = int(group_info[group]["size"])
            total += size
            category_counts[str(group_info[group]["category"])] += size
        return total, category_counts

    required = {category: math.ceil(float(fraction) * target_count) for category, fraction in minimum_fractions.items()}
    _, category_counts = counts()
    for category in sorted(required):
        deficit = required[category] - category_counts[category]
        if deficit <= 0:
            continue
        candidates = [
            group for group, assigned in assignments.items()
            if assigned == "train" and group_info[group]["category"] == category
        ]
        candidates.sort(key=lambda group: deterministic_key(seed + (1 if split_name == "test" else 2), group))
        for group in candidates:
            assignments[group] = split_name
            size = int(group_info[group]["size"])
            category_counts[category] += size
            deficit -= size
            if deficit <= 0:
                break
        if deficit > 0:
            raise PipelineError(f"Cannot satisfy {split_name} minimum fraction for category {category}")

    total, category_counts = counts()
    while total > target_count:
        candidates = []
        for group, assigned in assignments.items():
            if assigned != split_name:
                continue
            category = str(group_info[group]["category"])
            size = int(group_info[group]["size"])
            if category_counts[category] - size < required.get(category, 0):
                continue
            surplus = category_counts[category] - required.get(category, 0)
            candidates.append((-surplus, size, deterministic_key(seed, group), group))
        if not candidates:
            break
        _, size, _, group = min(candidates)
        category = str(group_info[group]["category"])
        assignments[group] = "train"
        category_counts[category] -= size
        total -= size


def _freeze_benchmark_manifests(
    *,
    test_records: list[dict[str, Any]],
    hard_hashes: set[str],
    config: dict[str, Any],
    root: Path,
    seed: int,
) -> dict[str, int]:
    benchmark = config["benchmark_sampling"]
    test_manifest = [{
        "content_hash": record["meta"]["content_hash"],
        "semantic_cluster_id": record["meta"]["semantic_cluster_id"],
        "category": record["meta"]["category"],
        "source": record["meta"]["source"],
        "hard": record["meta"]["content_hash"] in hard_hashes,
    } for record in test_records]
    atomic_write_jsonl(root / benchmark["frozen_manifest"], test_manifest)

    judge_config = benchmark["llm_judge"]
    target = min(int(judge_config["sample_size"]), len(test_records))
    hard_target = min(int(judge_config["include_hard_minimum"]), len(hard_hashes), target)
    hard_records = [
        record for record in test_records if str(record["meta"]["content_hash"]) in hard_hashes
    ]
    hard_records.sort(key=lambda record: deterministic_key(seed, str(record["meta"]["content_hash"])))
    selected = hard_records[:hard_target]
    selected_hashes = {str(record["meta"]["content_hash"]) for record in selected}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in test_records:
        content_hash = str(record["meta"]["content_hash"])
        if content_hash in selected_hashes:
            continue
        key = (str(record["meta"]["category"]), str(record["meta"]["source"]))
        buckets[key].append(record)
    for key in buckets:
        buckets[key].sort(key=lambda record: deterministic_key(seed, str(record["meta"]["content_hash"])))
    keys = sorted(buckets)
    while len(selected) < target and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < target:
                record = buckets[key].pop(0)
                selected.append(record)
                selected_hashes.add(str(record["meta"]["content_hash"]))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    judge_manifest = [{
        "content_hash": record["meta"]["content_hash"],
        "category": record["meta"]["category"],
        "source": record["meta"]["source"],
        "hard": record["meta"]["content_hash"] in hard_hashes,
    } for record in selected]
    atomic_write_jsonl(root / judge_config["manifest"], judge_manifest)
    return {"test": len(test_manifest), "judge": len(judge_manifest)}


def _write_datasheet(
    *, records_by_split: dict[str, list[dict[str, Any]]], root: Path,
    statistics: dict[str, Any], config: dict[str, Any],
) -> Path:
    all_records = [record for values in records_by_split.values() for record in values]
    licenses = Counter(str(record["meta"]["license"]) for record in all_records)
    sources = Counter(str(record["meta"]["source"]) for record in all_records)
    categories = Counter(str(record["meta"]["category"]) for record in all_records)
    missing_attribution = sum(
        1 for record in all_records
        if "CC BY" in str(record["meta"]["license"]).upper()
        and not record["meta"].get("attribution")
    )
    lines = [
        "# Datasheet: Docker/Kubernetes instruction tuning",
        "",
        "## Resumen",
        "",
        f"- Registros totales: {len(all_records)}",
        f"- Fecha de generacion (UTC): {statistics['created_at']}",
        f"- Tamano JSONL total: {sum(int(value['bytes']) for value in statistics['outputs'].values())} bytes",
        f"- Semilla de split: {config['seed']}",
        f"- Estrategia: {config['strategy']['algorithm']}",
        f"- Fugas exactas y semanticas: verificadas en {config['output']['leakage_report']}",
        f"- Registros CC BY/CC BY-SA sin atribucion: {missing_attribution}",
        f"- Estado: {'PROVISIONAL' if statistics.get('provisional') else 'FINAL'}",
        "",
        "## Splits",
        "",
        "| Split | Registros | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for split_name in ("train", "validation", "test"):
        output = statistics["outputs"][split_name]
        lines.append(f"| {split_name} | {output['count']} | {output['bytes']} | {output['sha256']} |")
    lines.extend(["", "## Fuentes", "", "| Fuente | Registros |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in sorted(sources.items()))
    lines.extend(["", "## Licencias", "", "| Licencia | URL canonica | Registros |", "|---|---|---:|"])
    lines.extend(
        f"| {name} | {_LICENSE_URLS.get(name, 'n/a')} | {count} |"
        for name, count in sorted(licenses.items())
    )
    lines.extend(["", "## Categorias", "", "| Categoria | Registros |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in sorted(categories.items()))
    lines.extend([
        "",
        "## Transformaciones",
        "",
        "Extraccion con revision de fuente, limpieza y redaccion de PII, normalizacion ChatML,",
        "deduplicacion exacta y semantica, validacion kubeconform/hadolint, split estratificado",
        "por categoria y fuente y auditoria exacta de similitud entre splits.",
        "",
        "Los datasets sin licencia confirmada permanecen en data/quarantine y no se incluyen.",
        "Las preguntas de Stack Overflow conservan URL, autor y atribucion CC BY-SA por registro.",
        "",
        "## Limitaciones",
        "",
        "- El contenido tecnico puede quedar obsoleto; la revision de fuente y fecha se conserva en meta.",
        "- La similitud semantica no demuestra equivalencia tecnica perfecta.",
        "- El set hard se construye con indicadores reproducibles de complejidad y debe revisarse junto al manifiesto.",
        "",
    ])
    destination = root / str(config["output"]["directory"]) / "DATASHEET.md"
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return destination


def run_split(
    *, config_path: Path = Path("config/splits.yaml"), root: Path = Path("."),
    skip_semantic_audit: bool = False, allow_provisional: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_yaml(root / config_path)
    validate_splits_config(config, require_approved=True)
    completeness = assess_pipeline_completeness(root=root)
    if not completeness["ready_for_final_split"] and not allow_provisional:
        codes = ", ".join(sorted({str(item["code"]) for item in completeness["issues"]}))
        raise PipelineError(
            f"Refusing final split because upstream lineage is incomplete: {codes}. "
            "Use --allow-provisional only for diagnostics."
        )
    input_path = root / config["input"]["path"]
    if not input_path.exists():
        raise PipelineError(f"Validated split input does not exist: {input_path}")
    records = list(read_jsonl(input_path))
    if not records:
        raise PipelineError("Split input is empty")
    for record in records:
        for field in config["input"]["required_fields"]:
            if field == "messages":
                if not record.get("messages"):
                    raise PipelineError("Split input record has no messages")
            else:
                _required_meta(record, str(field))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(_required_meta(record, "meta.semantic_cluster_id"))].append(record)
    group_info: dict[str, dict[str, Any]] = {}
    strata: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for group, members in groups.items():
        categories = {str(_required_meta(record, "meta.category")) for record in members}
        sources = {str(_required_meta(record, "meta.source")) for record in members}
        if len(categories) != 1 or len(sources) != 1:
            raise PipelineError(f"Semantic group {group} spans multiple source/category strata")
        category, source = next(iter(categories)), next(iter(sources))
        group_info[group] = {"category": category, "source": source, "size": len(members)}
        strata[(category, source)].append((group, len(members)))

    seed = int(config["seed"])
    ratios = config["ratios"]
    strategy = config["strategy"]
    assignments: dict[str, str] = {}
    for stratum, stratum_groups in sorted(strata.items()):
        allocated = allocate_stratum(
            stratum_groups,
            seed=seed + int(deterministic_key(seed, "|".join(stratum))[:8], 16),
            validation_ratio=float(ratios["validation"]),
            test_ratio=float(ratios["test"]),
            minimum_eval=int(strategy["minimum_records_per_stratum_per_eval_split"]),
            rare_threshold=int(strategy["rare_stratum"]["threshold"]),
        )
        assignments.update(allocated)

    total = len(records)
    minimum_fractions = {key: float(value) for key, value in config["category_constraints"]["minimum_fraction_each_eval_split"].items()}
    for split_name in ("validation", "test"):
        target = round(total * float(ratios[split_name]))
        _rebalance_categories(
            split_name=split_name,
            assignments=assignments,
            group_info=group_info,
            target_count=target,
            minimum_fractions=minimum_fractions,
            seed=seed,
        )

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for group, members in groups.items():
        records_by_split[assignments[group]].extend(members)
    for split_name in records_by_split:
        records_by_split[split_name].sort(
            key=lambda record: deterministic_key(seed, str(record.get("meta", {}).get("content_hash")))
        )

    if skip_semantic_audit:
        embedding_audit: dict[str, Any] = {"status": "skipped_provisionally"}
    else:
        audit_config = config["post_split_leakage_audit"]
        approximate_config = config["deduplication_before_split"]["approximate"]
        records_by_split, embedding_audit = audit_and_repair(
            records_by_split=records_by_split,
            embeddings_path=root / str(approximate_config["embeddings"]),
            pairs=[(str(pair[0]), str(pair[1])) for pair in audit_config["pairs"]],
            threshold=float(audit_config["threshold"]),
            seed=seed,
            max_iterations=int(audit_config["backfill_requirements"]["max_iterations"]),
        )
        for split_name in records_by_split:
            records_by_split[split_name].sort(
                key=lambda record: deterministic_key(seed, str(record.get("meta", {}).get("content_hash")))
            )

    required_categories = set(config["category_constraints"]["required_categories"])
    for split_name, split_records in records_by_split.items():
        present = {str(record["meta"]["category"]) for record in split_records}
        missing = required_categories - present
        if missing:
            raise PipelineError(f"{split_name} is missing required categories: {sorted(missing)}")
    tolerance = float(ratios["tolerance"])
    for split_name, expected in (("train", ratios["train"]), ("validation", ratios["validation"]), ("test", ratios["test"])):
        actual = len(records_by_split[split_name]) / total
        if abs(actual - float(expected)) > tolerance:
            raise PipelineError(f"{split_name} ratio {actual:.6f} is outside tolerance of {expected}")

    test_records = records_by_split["test"]
    hard_config = config["hard_test"]
    hard_candidates = []
    for record in test_records:
        if record["meta"]["category"] != hard_config["category"]:
            continue
        score, features = hard_complexity(record)
        if score < int(hard_config["complexity_score"]["min_score"]):
            continue
        explicit = bool(record.get("meta", {}).get("hard_example_is_true"))
        hard_candidates.append((not explicit, -score, deterministic_key(seed, record["meta"]["content_hash"]), record, features))
    hard_target = min(
        int(hard_config["maximum_records"]),
        max(int(hard_config["minimum_records"]), math.ceil(len(test_records) * float(hard_config["fraction_of_test"]))),
    )
    hard_candidates.sort(key=lambda item: item[:3])
    if len(hard_candidates) < hard_target:
        raise PipelineError(f"Only {len(hard_candidates)} hard test candidates available; {hard_target} required")
    hard_selected = hard_candidates[:hard_target]
    hard_sources = {str(item[3]["meta"]["source"]) for item in hard_selected}
    if len(hard_sources) < int(hard_config["source_diversity_minimum"]):
        raise PipelineError(f"Hard test has only {len(hard_sources)} sources")

    hashes = {split_name: {record["meta"]["content_hash"] for record in values} for split_name, values in records_by_split.items()}
    exact_overlaps = {
        f"{left}-{right}": len(hashes[left] & hashes[right])
        for left, right in (("train", "test"), ("train", "validation"), ("validation", "test"))
    }
    if any(exact_overlaps.values()):
        raise PipelineError(f"Exact cross-split leakage detected: {exact_overlaps}")
    cluster_sets = {
        split_name: {record["meta"]["semantic_cluster_id"] for record in values}
        for split_name, values in records_by_split.items()
    }
    group_overlaps = {
        f"{left}-{right}": len(cluster_sets[left] & cluster_sets[right])
        for left, right in (("train", "test"), ("train", "validation"), ("validation", "test"))
    }
    if any(group_overlaps.values()):
        raise PipelineError(f"Semantic group leakage detected: {group_overlaps}")

    output_dir = root / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for split_name, filename_key in (("train", "train"), ("validation", "validation"), ("test", "test")):
        split_path = output_dir / config["output"]["files"][filename_key]
        count, digest = atomic_write_jsonl(split_path, records_by_split[split_name])
        output_hashes[split_name] = {
            "count": count,
            "bytes": split_path.stat().st_size,
            "sha256": digest,
        }
    manifest = []
    for split_name, split_records in records_by_split.items():
        manifest.extend({
            "content_hash": record["meta"]["content_hash"],
            "semantic_cluster_id": record["meta"]["semantic_cluster_id"],
            "source": record["meta"]["source"],
            "category": record["meta"]["category"],
            "split": split_name,
        } for record in split_records)
    atomic_write_jsonl(root / config["output"]["assignment_manifest"], manifest)
    hard_manifest = [{
        "content_hash": item[3]["meta"]["content_hash"],
        "source": item[3]["meta"]["source"],
        "complexity_score": -item[1],
        "features": item[4],
    } for item in hard_selected]
    atomic_write_jsonl(root / config["output"]["hard_test_manifest"], hard_manifest)
    benchmark_manifests = _freeze_benchmark_manifests(
        test_records=test_records,
        hard_hashes={str(item["content_hash"]) for item in hard_manifest},
        config=config,
        root=root,
        seed=seed,
    )

    statistics = {
        "created_at": utc_now_iso(),
        "seed": seed,
        "total": total,
        "provisional": bool(skip_semantic_audit or not completeness["ready_for_final_split"]),
        "pipeline_completeness": completeness,
        "splits": split_statistics(records_by_split),
        "outputs": output_hashes,
        "hard_test": {"count": len(hard_manifest), "sources": sorted(hard_sources)},
        "benchmark_manifests": benchmark_manifests,
    }
    statistics["datasheet"] = str(
        _write_datasheet(
            records_by_split=records_by_split,
            root=root,
            statistics=statistics,
            config=config,
        ).relative_to(root)
    )
    atomic_write_json(root / config["output"]["statistics"], statistics)
    leakage = {
        "exact_overlap": exact_overlaps,
        "semantic_group_overlap": group_overlaps,
        "embedding_audit": embedding_audit,
    }
    atomic_write_json(root / config["output"]["leakage_report"], leakage)
    return statistics
