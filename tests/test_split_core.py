from __future__ import annotations

from docker_k8s_finetune.split.core import allocate_stratum, hard_complexity


def test_allocate_stratum_is_deterministic_and_group_safe() -> None:
    groups = [(f"g-{index}", 1) for index in range(100)]
    first = allocate_stratum(groups, seed=3407, validation_ratio=0.05, test_ratio=0.05, minimum_eval=2, rare_threshold=40)
    second = allocate_stratum(list(reversed(groups)), seed=3407, validation_ratio=0.05, test_ratio=0.05, minimum_eval=2, rare_threshold=40)
    assert first == second
    assert list(first.values()).count("test") == 5
    assert list(first.values()).count("validation") == 5


def test_hard_complexity_detects_multifactor_incident() -> None:
    record = {"messages": [
        {"role": "user", "content": "CoreDNS and kubelet fail intermittently after an OOM event; first inspect logs, then diagnose CNI."},
        {"role": "assistant", "content": "Check memory limits, DNS endpoints and containerd logs."},
    ]}
    score, features = hard_complexity(record)
    assert score >= 5
    assert "multiple_components" in features
