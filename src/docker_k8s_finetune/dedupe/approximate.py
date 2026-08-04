from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import load_yaml
from ..errors import PipelineError
from ..io import atomic_write_json, atomic_write_jsonl, content_hash, read_jsonl


def semantic_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")
    by_role = {item.get("role"): item.get("content") for item in messages if isinstance(item, dict)}
    user = str(by_role.get("user", "")).strip()
    assistant = str(by_role.get("assistant", "")).strip()
    if not user or not assistant:
        raise ValueError("record lacks user or assistant content")
    return f"{user}\n\n{assistant}"


def connected_components_from_neighbors(
    neighbors: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[int, list[int]]:
    count = int(neighbors.shape[0])
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for index in range(count):
        for neighbor, score in zip(neighbors[index], scores[index]):
            other = int(neighbor)
            if other < 0 or other == index or float(score) < threshold:
                continue
            union(index, other)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        components[find(index)].append(index)
    return dict(components)


def run_approximate_dedupe(
    *, config_path: Path = Path("config/dedupe.yaml"), root: Path = Path(".")
) -> dict[str, Any]:
    try:
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PipelineError("Install the dedupe extra before approximate deduplication: pip install -e .[dedupe]") from exc

    root = root.resolve()
    config = load_yaml(root / config_path)
    approximate = config["approximate"]
    input_path = root / config["exact"]["output"]
    if not input_path.exists():
        raise PipelineError(f"Exact-deduplicated input does not exist: {input_path}")
    records = list(read_jsonl(input_path))
    if not records:
        raise PipelineError("Exact-deduplicated input is empty")
    texts = [semantic_text(record) for record in records]
    requested_device = str(approximate.get("device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    revision = approximate.get("model_revision") or None
    model = SentenceTransformer(str(approximate["model"]), device=device, revision=revision)
    embeddings = model.encode(
        texts,
        batch_size=int(approximate["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)

    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(
        embeddings.shape[1], int(approximate["hnsw_m"]), faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = int(approximate["hnsw_ef_construction"])
    index.hnsw.efSearch = int(approximate["hnsw_ef_search"])
    index.add(embeddings)
    top_k = min(int(approximate["top_k"]), len(records))
    scores, neighbors = index.search(embeddings, top_k)
    components = connected_components_from_neighbors(neighbors, scores, float(approximate["threshold"]))

    tiers = approximate.get("source_tiers", {})
    keep_indices: set[int] = set()
    duplicates: list[dict[str, Any]] = []
    removed_by_source: Counter[str] = Counter()
    cluster_sizes: Counter[int] = Counter()
    for members in components.values():
        cluster_sizes[len(members)] += 1

        def rank(index_value: int) -> tuple[float, int, str]:
            meta = records[index_value].get("meta", {})
            quality = float(meta.get("quality_score", 0.0))
            tier = int(tiers.get(str(meta.get("source")), 99))
            return (-quality, tier, str(meta.get("content_hash", "")))

        representative = min(members, key=rank)
        hashes = sorted(str(records[index_value].get("meta", {}).get("content_hash", "")) for index_value in members)
        cluster_id = content_hash("|".join(hashes))
        records[representative].setdefault("meta", {})["semantic_cluster_id"] = cluster_id
        keep_indices.add(representative)
        for index_value in members:
            if index_value == representative:
                continue
            meta = records[index_value].get("meta", {})
            source = str(meta.get("source", "unknown"))
            removed_by_source[source] += 1
            similarity = float(np.dot(embeddings[index_value], embeddings[representative]))
            duplicates.append({
                "semantic_cluster_id": cluster_id,
                "source": source,
                "source_record_id": meta.get("source_record_id"),
                "kept_source": records[representative].get("meta", {}).get("source"),
                "kept_source_record_id": records[representative].get("meta", {}).get("source_record_id"),
                "representative_similarity": round(similarity, 8),
            })

    kept_order = sorted(keep_indices)
    kept_records = (records[index_value] for index_value in kept_order)
    output_path = root / approximate["output"]
    duplicates_path = root / approximate["duplicates"]
    kept, output_hash = atomic_write_jsonl(output_path, kept_records)
    removed, duplicates_hash = atomic_write_jsonl(duplicates_path, duplicates)
    embeddings_path = root / approximate["embeddings"]
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_embeddings = embeddings_path.with_name(f".{embeddings_path.name}.tmp.npz")
    np.savez_compressed(
        temporary_embeddings,
        embeddings=embeddings[kept_order],
        content_hashes=np.asarray(
            [str(records[index_value]["meta"]["content_hash"]) for index_value in kept_order],
            dtype=str,
        ),
    )
    os.replace(temporary_embeddings, embeddings_path)
    embeddings_hash = hashlib.sha256(embeddings_path.read_bytes()).hexdigest()
    report = {
        "version": int(config["version"]),
        "method": "sentence_transformer_faiss_hnsw_connected_components",
        "model": approximate["model"],
        "model_revision": revision,
        "device": device,
        "threshold": float(approximate["threshold"]),
        "top_k": top_k,
        "counts": {"input": len(records), "kept": kept, "removed": removed, "clusters": len(components)},
        "cluster_size_histogram": {str(key): value for key, value in sorted(cluster_sizes.items())},
        "removed_by_source": dict(sorted(removed_by_source.items())),
        "output": {"path": approximate["output"], "sha256": output_hash},
        "embeddings": {"path": approximate["embeddings"], "sha256": embeddings_hash},
        "duplicates": {"path": approximate["duplicates"], "sha256": duplicates_hash},
        "limitations": "HNSW is an approximate neighbor index; post-split leakage audit remains mandatory.",
    }
    atomic_write_json(root / approximate["report"], report)
    return report
