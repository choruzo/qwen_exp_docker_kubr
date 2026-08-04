from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..errors import PipelineError
from .core import deterministic_key, hard_complexity


@dataclass(frozen=True)
class CrossSplitLeak:
    query_hash: str
    reference_hash: str
    query_split: str
    reference_split: str
    similarity: float


def load_embedding_cache(path: Path) -> tuple[dict[str, int], np.ndarray]:
    if not path.is_file():
        raise PipelineError(f"Semantic embedding cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        hashes = payload["content_hashes"].astype(str)
        embeddings = payload["embeddings"].astype("float32", copy=False)
    if embeddings.ndim != 2 or len(hashes) != embeddings.shape[0]:
        raise PipelineError(f"Invalid semantic embedding cache: {path}")
    mapping = {value: index for index, value in enumerate(hashes.tolist())}
    if len(mapping) != len(hashes):
        raise PipelineError("Semantic embedding cache contains duplicate content hashes")
    return mapping, embeddings


def find_cross_split_leaks(
    *,
    hashes: list[str],
    vectors: np.ndarray,
    assignments: Mapping[str, str],
    pairs: list[tuple[str, str]],
    threshold: float,
    block_size: int = 256,
) -> list[CrossSplitLeak]:
    index_by_split: dict[str, list[int]] = defaultdict(list)
    for index, content_hash in enumerate(hashes):
        index_by_split[assignments[content_hash]].append(index)
    leaks: list[CrossSplitLeak] = []
    for left, right in pairs:
        if left == "train":
            query_split, reference_split = right, left
        elif right == "train":
            query_split, reference_split = left, right
        else:
            query_split, reference_split = right, left
        query_indices = np.asarray(index_by_split[query_split], dtype=np.int64)
        reference_indices = np.asarray(index_by_split[reference_split], dtype=np.int64)
        if not len(query_indices) or not len(reference_indices):
            continue
        references = vectors[reference_indices]
        for start in range(0, len(query_indices), block_size):
            block_indices = query_indices[start : start + block_size]
            scores = vectors[block_indices] @ references.T
            best_columns = np.argmax(scores, axis=1)
            best_scores = scores[np.arange(len(block_indices)), best_columns]
            for row, score in enumerate(best_scores):
                if float(score) < threshold:
                    continue
                query_index = int(block_indices[row])
                reference_index = int(reference_indices[int(best_columns[row])])
                leaks.append(CrossSplitLeak(
                    query_hash=hashes[query_index],
                    reference_hash=hashes[reference_index],
                    query_split=query_split,
                    reference_split=reference_split,
                    similarity=float(score),
                ))
    return leaks


def audit_and_repair(
    *,
    records_by_split: dict[str, list[dict[str, Any]]],
    embeddings_path: Path,
    pairs: list[tuple[str, str]],
    threshold: float,
    seed: int,
    max_iterations: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cache_indices, cache_vectors = load_embedding_cache(embeddings_path)
    records: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for split_name, split_records in records_by_split.items():
        for record in split_records:
            content_hash = str(record["meta"]["content_hash"])
            if content_hash not in cache_indices:
                raise PipelineError(f"Missing cached embedding for {content_hash}")
            records[content_hash] = record
            assignments[content_hash] = split_name
            groups[str(record["meta"]["semantic_cluster_id"])].append(content_hash)
    hashes = sorted(records)
    hash_positions = {value: index for index, value in enumerate(hashes)}
    vectors = np.stack([cache_vectors[cache_indices[value]] for value in hashes]).astype("float32", copy=False)
    events: list[dict[str, Any]] = []
    initial_leaks: int | None = None

    def group_for(content_hash: str) -> str:
        return str(records[content_hash]["meta"]["semantic_cluster_id"])

    def group_assignment(group: str) -> str:
        values = {assignments[value] for value in groups[group]}
        if len(values) != 1:
            raise PipelineError(f"Semantic group {group} was split during leakage repair")
        return next(iter(values))

    def candidate_is_safe(group: str, target: str) -> bool:
        member_indices = [hash_positions[value] for value in groups[group]]
        other_indices = [
            index for index, value in enumerate(hashes)
            if group_for(value) != group and assignments[value] != target
        ]
        if not other_indices:
            return True
        scores = vectors[member_indices] @ vectors[other_indices].T
        return float(np.max(scores)) < threshold

    blocked_groups: set[str] = set()
    for iteration in range(1, max_iterations + 1):
        leaks = find_cross_split_leaks(
            hashes=hashes,
            vectors=vectors,
            assignments=assignments,
            pairs=pairs,
            threshold=threshold,
        )
        if initial_leaks is None:
            initial_leaks = len(leaks)
        if not leaks:
            repaired = {"train": [], "validation": [], "test": []}
            for content_hash in hashes:
                repaired[assignments[content_hash]].append(records[content_hash])
            return repaired, {
                "status": "clean",
                "threshold": threshold,
                "initial_leaks": initial_leaks,
                "final_leaks": 0,
                "iterations": iteration - 1,
                "events": events,
            }

        deficits: list[tuple[str, str, str, int]] = []
        groups_to_move = sorted({group_for(leak.query_hash) for leak in leaks})
        for group in groups_to_move:
            previous = group_assignment(group)
            if previous == "train":
                continue
            members = groups[group]
            exemplar = records[members[0]]["meta"]
            for content_hash in members:
                assignments[content_hash] = "train"
            blocked_groups.add(group)
            deficits.append((previous, str(exemplar["category"]), str(exemplar["source"]), len(members)))
            events.append({"iteration": iteration, "action": "move_leak_to_train", "group": group, "from": previous, "count": len(members)})

        for target, category, source, needed in deficits:
            candidates = []
            for group, members in groups.items():
                if group in blocked_groups or group_assignment(group) != "train":
                    continue
                meta = records[members[0]]["meta"]
                if str(meta["category"]) != category or str(meta["source"]) != source:
                    continue
                if len(members) > needed or not candidate_is_safe(group, target):
                    continue
                hard_score = hard_complexity(records[members[0]])[0] if target == "test" and category == "troubleshooting" else 0
                candidates.append((-hard_score, deterministic_key(seed + iteration, group), group))
            candidates.sort()
            remaining = needed
            for _, _, group in candidates:
                members = groups[group]
                for content_hash in members:
                    assignments[content_hash] = target
                remaining -= len(members)
                events.append({"iteration": iteration, "action": "backfill", "group": group, "to": target, "count": len(members)})
                if remaining == 0:
                    break
            if remaining:
                raise PipelineError(
                    f"Cannot backfill {needed} {target} records for stratum {category}/{source} after leakage repair"
                )
    raise PipelineError(f"Embedding leakage audit did not converge after {max_iterations} iterations")
