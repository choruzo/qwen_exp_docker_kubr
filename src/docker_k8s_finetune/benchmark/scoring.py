from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import load_yaml
from ..errors import PipelineError
from ..validate.core import extract_dockerfile, extract_yaml
from ..validate.pipeline import _validate_dockerfiles, _validate_kubernetes
from .core import parse_judge_payload


def _verify_judge_props(
    props: Mapping[str, Any], *, expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    alias = str(expected_identity["alias"])
    if str(props.get("model_alias")) != alias:
        raise PipelineError(
            f"Judge llama-server alias mismatch: {props.get('model_alias')} != {alias}"
        )
    served_raw = str(props.get("model_path", "")).strip()
    served_name = served_raw.replace("\\", "/").rsplit("/", 1)[-1]
    expected_name = str(expected_identity["filename"])
    if served_name.casefold() != expected_name.casefold():
        raise PipelineError(
            f"Judge llama-server model mismatch: {served_name} != {expected_name}"
        )
    quantization = str(expected_identity["quantization"])
    normalized_quantization = quantization.replace("_", "").replace("-", "").casefold()
    normalized_name = served_name.replace("_", "").replace("-", "").casefold()
    if normalized_quantization not in normalized_name:
        raise PipelineError(
            f"Judge llama-server quantization mismatch: {served_name} lacks {quantization}"
        )
    return {
        "alias": alias,
        "server_model_path": served_raw,
        "filename": expected_name,
        "quantization": quantization,
        "ftype": props.get("model_ftype"),
    }


def semantic_scores(
    references: list[str], predictions: list[str], config: Mapping[str, Any],
) -> list[float]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PipelineError("Install the dedupe extra for semantic benchmark scoring") from exc
    requested = str(config.get("device", "auto"))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    model = SentenceTransformer(
        str(config["model"]),
        revision=str(config["model_revision"]),
        device=device,
    )
    vectors = model.encode(
        references + predictions,
        batch_size=int(config["batch_size"]),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32", copy=False)
    size = len(references)
    return np.sum(vectors[:size] * vectors[size:], axis=1).astype(float).tolist()


def syntax_scores(
    records: list[dict[str, Any]], *, variant: str, benchmark_config: Mapping[str, Any], root: Path,
) -> dict[str, dict[str, Any]]:
    validation = load_yaml(root / str(benchmark_config["syntax_validation"]["config"]))
    work = root / str(benchmark_config["outputs"]["work_dir"]) / variant / "syntax"
    if work.exists():
        resolved = work.resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise PipelineError(f"Unsafe benchmark work directory: {resolved}")
        shutil.rmtree(resolved)
    yaml_dir = work / "yaml"
    dockerfile_dir = work / "dockerfile"
    yaml_dir.mkdir(parents=True)
    dockerfile_dir.mkdir(parents=True)
    yaml_files: list[Path] = []
    dockerfiles: list[Path] = []
    identifiers: dict[str, str] = {}
    scores: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        category = str(record["category"])
        if category not in {"generacion_yaml", "dockerfile"} or record.get("error"):
            continue
        content_hash = str(record["content_hash"])
        prediction = str(record["prediction"])
        if category == "generacion_yaml":
            content = extract_yaml(prediction)
            if content is None:
                scores[content_hash] = {"valid": 0.0, "reason": "yaml_block_missing"}
                continue
            path = yaml_dir / f"{index:06d}-{content_hash[:16]}.yaml"
            path.write_text(content + "\n", encoding="utf-8", newline="\n")
            yaml_files.append(path)
        else:
            content = extract_dockerfile(prediction)
            if content is None:
                scores[content_hash] = {"valid": 0.0, "reason": "dockerfile_block_missing"}
                continue
            path = dockerfile_dir / f"{index:06d}-{content_hash[:16]}.Dockerfile"
            path.write_text(content + "\n", encoding="utf-8", newline="\n")
            dockerfiles.append(path)
        identifiers[path.name] = content_hash
    failures = _validate_kubernetes(work, yaml_files, validation)
    failures.update(_validate_dockerfiles(work, dockerfiles, validation))
    for filename, content_hash in identifiers.items():
        reason = failures.get(filename)
        scores[content_hash] = {"valid": 0.0 if reason else 1.0, "reason": reason}
    return scores


class JudgeClient:
    def __init__(self, config: Mapping[str, Any], prompt: str) -> None:
        self.config = config
        self.prompt = prompt
        self.base_url = os.environ.get(str(config["base_url_env"]), "").rstrip("/")
        self.api_key = os.environ.get(str(config["api_key_env"]), "")
        self.model = os.environ.get(str(config["model_env"]), "")
        if not self.base_url or not self.model:
            raise PipelineError(f"Set {config['base_url_env']} and {config['model_env']} for LLM judging")
        identity = config.get("model_identity")
        self.provenance: dict[str, Any] | None = None
        if isinstance(identity, Mapping):
            expected = dict(identity)
            if self.model != str(expected["alias"]):
                raise PipelineError(
                    f"Judge model alias differs from fixed identity: {self.model} != {expected['alias']}"
                )
            self.provenance = {"expected": expected}
            if config.get("require_llama_props"):
                import requests

                server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
                try:
                    response = requests.get(
                        f"{server_root}/props",
                        timeout=float(config.get("identity_timeout_seconds", 10)),
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise PipelineError(f"Cannot verify judge llama-server identity: {exc}") from exc
                self.provenance["served"] = _verify_judge_props(
                    response.json(), expected_identity=expected
                )

    def score(self, *, question: str, reference: str, candidate: str) -> dict[str, Any]:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        user = (
            f"PREGUNTA:\n{question}\n\nRESPUESTA DE REFERENCIA:\n{reference}"
            f"\n\nRESPUESTA CANDIDATA:\n{candidate}"
        )
        payload = {
            "model": self.model,
            "seed": int(self.config["seed"]),
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.config["temperature"]),
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(int(self.config["max_retries"])):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=float(self.config["timeout_seconds"]),
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return parse_judge_payload(json.loads(content))
            except (requests.RequestException, KeyError, IndexError, TypeError, json.JSONDecodeError, PipelineError) as exc:
                last_error = exc
                if attempt + 1 < int(self.config["max_retries"]):
                    time.sleep(min(2 ** attempt, 4))
        raise PipelineError(f"LLM judge failed after retries: {last_error}")
