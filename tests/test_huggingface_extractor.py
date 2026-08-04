from docker_k8s_finetune.extract.huggingface import HuggingFaceExtractor, normalize_licenses


ROOT = {
    "project": {"quarantine_root": "data/quarantine"},
    "defaults": {},
    "license_policy": {"the_stack_permissive_allowlist": ["Apache-2.0", "MIT"]},
    "category_rules": {"priority": [], "fallback": "concepto"},
}


def test_normalize_licenses_handles_aliases_and_nested_values() -> None:
    assert normalize_licenses(["mit", {"spdx_id": "apache-2.0"}]) == ["MIT", "Apache-2.0"]


def test_per_record_license_rejects_disallowed_rows() -> None:
    source = {
        "kind": "huggingface_dataset",
        "tier": 2,
        "dataset": "example/data",
        "license": "per_record",
        "license_fields": ["licenses"],
        "per_record_allowlist_ref": "license_policy.the_stack_permissive_allowlist",
        "content_field": "content",
        "category": "generacion_yaml",
        "normalization": "reverse_instruction",
        "sampling": {"strategy": "deterministic_hash", "max_records": 10},
    }
    extractor = HuggingFaceExtractor(source_name="stack", source_config=source, root_config=ROOT)
    selected, rejected = extractor._selected_records(
        [
            {"content": "apiVersion: v1\nkind: Pod", "licenses": ["mit"]},
            {"content": "apiVersion: v1\nkind: Service", "licenses": ["gpl-3.0"]},
        ],
        "abc123",
    )
    assert [record.license for record in selected] == ["MIT"]
    assert len(rejected) == 1


def test_direct_pair_is_preserved_in_metadata() -> None:
    source = {
        "kind": "huggingface_dataset",
        "tier": 2,
        "dataset": "example/data",
        "license": "Apache-2.0",
        "field_map": {"user": "instruction", "assistant": "output"},
        "category": "comando_cli",
        "normalization": "direct_pair",
    }
    extractor = HuggingFaceExtractor(source_name="pairs", source_config=source, root_config=ROOT)
    selected, rejected = extractor._selected_records(
        [{"instruction": "List pods", "output": "kubectl get pods"}], "abc123"
    )
    assert not rejected
    assert selected[0].metadata["pair"]["assistant"] == "kubectl get pods"


def test_quarantine_content_fields_are_preserved_as_json() -> None:
    source = {
        "kind": "huggingface_dataset",
        "tier": "quarantine",
        "dataset": "example/data",
        "license": "unknown",
        "normalization": "none",
        "content_fields": ["question", "command", "chain_of_thought"],
    }
    extractor = HuggingFaceExtractor(source_name="unknown", source_config=source, root_config=ROOT)
    selected, rejected = extractor._selected_records(
        [{"question": "List pods", "command": "kubectl get pods", "chain_of_thought": None}], "abc123"
    )
    assert not rejected
    assert '"command": "kubectl get pods"' in selected[0].raw_content


def test_dataset_viewer_fallback_paginates(monkeypatch) -> None:
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, *, params, timeout):
        calls.append(params.copy())
        count = min(params["length"], 150 - params["offset"])
        return Response({
            "rows": [{"row": {"Question": f"q-{index}"}} for index in range(params["offset"], params["offset"] + count)],
            "num_rows_total": 150,
        })

    monkeypatch.setattr("docker_k8s_finetune.extract.huggingface.requests.get", fake_get)
    source = {
        "kind": "huggingface_dataset",
        "tier": 2,
        "dataset": "example/data",
        "split": "train",
        "viewer_config": "default",
        "max_records": 150,
        "license": "CC-BY-4.0",
    }
    extractor = HuggingFaceExtractor(source_name="viewer", source_config=source, root_config=ROOT)
    rows = list(extractor._viewer_rows())
    assert len(rows) == 150
    assert [call["offset"] for call in calls] == [0, 100]
