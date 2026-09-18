from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import load_yaml
from ..errors import PipelineError
from ..train.core import verify_export_artifact, verify_local_model_provenance


def _completion_length_through_first_eos(
    token_ids: list[int], eos_token_ids: set[int],
) -> int:
    """Count generated tokens through EOS, excluding batch padding after it."""
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return index + 1
    return len(token_ids)


def _cuda_runtime_metadata(torch: Any) -> dict[str, Any]:
    """Return the concrete CUDA runtime used for an in-process benchmark."""
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    major = int(getattr(properties, "major", 0))
    minor = int(getattr(properties, "minor", 0))
    runtime = {
        "device": str(properties.name),
        "device_index": index,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
    }
    hip = getattr(torch.version, "hip", None)
    if hip:
        runtime.update({
            "accelerator": "rocm",
            "gcn_arch": str(getattr(properties, "gcnArchName", "")),
            "torch_hip_version": str(hip),
        })
    else:
        runtime.update({
            "accelerator": "cuda",
            "compute_capability": f"{major}.{minor}",
            "torch_cuda_version": str(torch.version.cuda),
        })
    return runtime


def _verify_llama_props(
    props: Mapping[str, Any], *, expected_alias: str, expected_quantization: str,
    root: Path, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if str(props.get("model_alias")) != expected_alias:
        raise PipelineError(
            f"llama-server alias mismatch: {props.get('model_alias')} != {expected_alias}"
        )
    served_raw = str(props.get("model_path", "")).strip()
    if not served_raw:
        raise PipelineError("llama-server did not report model_path")
    # llama-server runs on the Windows host while this verifier normally runs
    # in the Linux training container. Compare normalized complete manifest
    # suffixes so D:\repo\artifacts\... and /workspace/project/artifacts/...
    # identify the same bind-mounted artifact without weakening this to a
    # basename-only match.
    served_normalized = served_raw.replace("\\", "/").rstrip("/").casefold()
    matches = [
        entry for entry in provenance.get("files", [])
        if served_normalized.endswith(
            "/" + str(entry["path"]).replace("\\", "/").strip("/").casefold()
        )
        or served_normalized
        == str(entry["path"]).replace("\\", "/").strip("/").casefold()
    ]
    if len(matches) != 1:
        raise PipelineError(f"llama-server model is not the manifested GGUF: {served_raw}")
    normalized = expected_quantization.replace("_", "").replace("-", "").lower()
    served_name = served_normalized.rsplit("/", 1)[-1].replace("_", "").replace("-", "")
    if normalized not in served_name:
        raise PipelineError(
            f"llama-server GGUF quantization mismatch: {served_raw} lacks {expected_quantization}"
        )
    return {
        "alias": expected_alias,
        "model_path": str(matches[0]["path"]),
        "server_model_path": served_raw,
        "sha256": matches[0]["sha256"],
        "quantization": expected_quantization,
        "ftype": props.get("model_ftype"),
    }


@dataclass(frozen=True)
class Generation:
    text: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    batch_size: int = 1
    batch_completion_tokens_total: int | None = None
    finished_eos: bool | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.latency_seconds <= 0:
            return None
        return self.completion_tokens / self.latency_seconds

    @property
    def batch_tokens_per_second(self) -> float | None:
        total = self.batch_completion_tokens_total
        if not total or self.latency_seconds <= 0:
            return None
        return total / self.latency_seconds


class OpenAICompatibleBackend:
    def __init__(self, config: Mapping[str, Any], root: Path) -> None:
        self.base_url = os.environ.get(str(config["base_url_env"]), "").rstrip("/")
        self.api_key = os.environ.get(str(config["api_key_env"]), "")
        self.model = os.environ.get(str(config["model_env"]), "")
        if not self.base_url or not self.model:
            raise PipelineError(
                f"Set {config['base_url_env']} and {config['model_env']} for the OpenAI-compatible backend"
            )
        self.provenance = None
        self.runtime: dict[str, Any] = {
            "backend": "openai_compatible",
            "model": self.model,
        }
        if config.get("export_manifest"):
            self.provenance = verify_export_artifact(
                root, Path(str(config["export_manifest"])), str(config["artifact"])
            )
        if config.get("require_llama_props"):
            if self.provenance is None:
                raise PipelineError("llama-server identity verification requires an export manifest")
            import requests

            server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            response = requests.get(
                f"{server_root}/props",
                timeout=float(config.get("identity_timeout_seconds", 10)),
            )
            response.raise_for_status()
            self.provenance["served_model"] = _verify_llama_props(
                response.json(),
                expected_alias=self.model,
                expected_quantization=str(config["expected_quantization"]),
                root=root,
                provenance=self.provenance,
            )
            self.runtime["served_model"] = dict(self.provenance["served_model"])

    def generate(self, messages: list[dict[str, str]], generation: Mapping[str, Any]) -> Generation:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "seed": int(generation["seed"]),
            "messages": messages,
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "max_tokens": int(generation["max_new_tokens"]),
            "repeat_penalty": float(generation["repetition_penalty"]),
            "stream": False,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(generation.get("timeout_seconds", 300)),
        )
        latency = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        try:
            choice = body["choices"][0]
            text = str(choice["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise PipelineError(f"Invalid generation response: {body}") from exc
        usage = body.get("usage", {})
        return Generation(
            text=text,
            latency_seconds=latency,
            prompt_tokens=int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None,
            completion_tokens=int(usage["completion_tokens"]) if usage.get("completion_tokens") is not None else None,
            finished_eos=str(choice.get("finish_reason", "")).casefold() == "stop",
        )


class UnslothBackend:
    def __init__(self, variant: Mapping[str, Any], root: Path) -> None:
        try:
            import torch
            from unsloth import FastVisionModel
        except ImportError as exc:
            raise PipelineError("Unsloth benchmark backend must run in compose.train.yaml") from exc
        if not torch.cuda.is_available():
            raise PipelineError("A PyTorch GPU is required for the Unsloth benchmark backend")
        model_path = root / str(variant["model_path"])
        if not model_path.is_dir():
            raise PipelineError(f"Benchmark model path does not exist: {model_path}")
        self.provenance = None
        provenance_config = variant.get("provenance_config")
        if provenance_config:
            training = load_yaml(root / str(provenance_config))
            self.provenance = verify_local_model_provenance(model_path, training["model"])
        elif variant.get("export_manifest"):
            self.provenance = verify_export_artifact(
                root, Path(str(variant["export_manifest"])), str(variant["artifact"])
            )
        self.torch = torch
        self.runtime = {
            "backend": "unsloth",
            **_cuda_runtime_metadata(torch),
        }
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=str(model_path),
            load_in_4bit=bool(variant.get("load_in_4bit", False)),
            dtype=torch.bfloat16,
        )
        FastVisionModel.for_inference(self.model)

    def generate(self, messages: list[dict[str, str]], generation: Mapping[str, Any]) -> Generation:
        return self.generate_batch([messages], generation)[0]

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        generation: Mapping[str, Any],
    ) -> list[Generation]:
        if not messages_batch:
            return []
        if (
            len(messages_batch) > 1
            and generation.get("batch_policy") != "left_padding_first_eos_v1"
        ):
            raise PipelineError(
                "Batched Unsloth generation requires batch_policy=left_padding_first_eos_v1"
            )
        self.torch.manual_seed(int(generation["seed"]))
        self.torch.cuda.manual_seed_all(int(generation["seed"]))
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(generation.get("enable_thinking", False)),
            )
            for messages in messages_batch
        ]
        # Decoder-only batched generation must be left padded so the final
        # non-padding token is the generation boundary for every prompt.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                do_sample=float(generation["temperature"]) > 0,
                temperature=max(float(generation["temperature"]), 1e-5),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                use_cache=True,
            )
        self.torch.cuda.synchronize()
        latency = time.perf_counter() - started
        padded_prompt_tokens = int(inputs["input_ids"].shape[-1])
        prompt_token_counts = [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
        padded_completions = [row[padded_prompt_tokens:] for row in output]
        eos_value = self.tokenizer.eos_token_id
        eos_token_ids = {
            int(value)
            for value in (eos_value if isinstance(eos_value, (list, tuple, set)) else [eos_value])
            if value is not None
        }
        completion_token_lists = [
            [int(token_id) for token_id in completion.tolist()]
            for completion in padded_completions
        ]
        completion_finished_eos = [
            any(token_id in eos_token_ids for token_id in token_ids)
            for token_ids in completion_token_lists
        ]
        completion_token_counts = [
            _completion_length_through_first_eos(token_ids, eos_token_ids)
            for token_ids in completion_token_lists
        ]
        completions = [
            completion[:completion_tokens]
            for completion, completion_tokens in zip(
                padded_completions, completion_token_counts
            )
        ]
        batch_completion_tokens = sum(completion_token_counts)
        return [
            Generation(
                text=self.tokenizer.decode(completion, skip_special_tokens=True).strip(),
                latency_seconds=latency,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                batch_size=len(messages_batch),
                batch_completion_tokens_total=batch_completion_tokens,
                finished_eos=finished_eos,
            )
            for completion, prompt_tokens, completion_tokens, finished_eos in zip(
                completions,
                prompt_token_counts,
                completion_token_counts,
                completion_finished_eos,
            )
        ]


def build_backend(variant: Mapping[str, Any], root: Path) -> Any:
    backend = str(variant["backend"])
    if backend == "unsloth":
        return UnslothBackend(variant, root)
    if backend == "openai_compatible":
        return OpenAICompatibleBackend(variant, root)
    raise PipelineError(f"Unsupported benchmark backend: {backend}")
