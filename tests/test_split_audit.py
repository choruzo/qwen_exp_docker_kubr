from __future__ import annotations

import numpy as np

from docker_k8s_finetune.split.audit import find_cross_split_leaks


def test_exact_cross_split_audit_finds_nearest_leak() -> None:
    hashes = ["train-a", "train-b", "test-a", "val-a"]
    vectors = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.99, 0.01],
        [-1.0, 0.0],
    ], dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assignments = {"train-a": "train", "train-b": "train", "test-a": "test", "val-a": "validation"}
    leaks = find_cross_split_leaks(
        hashes=hashes,
        vectors=vectors,
        assignments=assignments,
        pairs=[("train", "test"), ("train", "validation"), ("validation", "test")],
        threshold=0.92,
        block_size=2,
    )
    assert [(item.query_hash, item.reference_hash) for item in leaks] == [("test-a", "train-a")]
