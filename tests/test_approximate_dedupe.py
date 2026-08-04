from __future__ import annotations

import numpy as np

from docker_k8s_finetune.dedupe.approximate import connected_components_from_neighbors, semantic_text


def test_connected_components_apply_threshold_transitively() -> None:
    neighbors = np.array([[0, 1], [1, 2], [2, 1], [3, -1]])
    scores = np.array([[1.0, 0.95], [1.0, 0.93], [1.0, 0.93], [1.0, -1.0]])
    components = connected_components_from_neighbors(neighbors, scores, 0.92)
    assert sorted(sorted(values) for values in components.values()) == [[0, 1, 2], [3]]


def test_semantic_text_excludes_system() -> None:
    record = {"messages": [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]}
    assert semantic_text(record) == "question\n\nanswer"
