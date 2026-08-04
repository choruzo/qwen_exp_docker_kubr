from docker_k8s_finetune.extract.runner import _selected_names


def test_quarantine_sources_are_opt_in() -> None:
    sources = {
        "safe": {"tier": 1, "enabled": True},
        "quarantine": {"tier": "quarantine", "enabled": True},
    }
    assert _selected_names(sources, None, False) == ["safe"]
    assert _selected_names(sources, None, True) == ["safe", "quarantine"]
