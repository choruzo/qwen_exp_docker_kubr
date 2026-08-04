from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..errors import PipelineError


@dataclass(frozen=True)
class Generation:
    text: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.completion_tokens or self.latency_seconds <= 0:
            return None
        return self.completion_tokens / self.latency_seconds


class OpenAICompatibleBackend:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.base_url = os.environ.get(str(config["base_url_env"]), "").rstrip("/")
        self.api_key = os.environ.get(str(config["api_key_env"]), "")
        self.model = os.environ.get(str(config["model_env"]), "")
        if not self.base_url or not self.model:
            raise PipelineError(
                f"Set {config['base_url_env']} and {config['model_env']} for the OpenAI-compatible backend"
            )

    def generate(self, messages: list[dict[str, str]], generation: Mapping[str, Any]) -> Generation:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "max_tokens": int(generation["max_new_tokens"]),
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
            text = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise PipelineError(f"Invalid generation response: {body}") from exc
        usage = body.get("usage", {})
        return Generation(
            text=text,
            latency_seconds=latency,
            prompt_tokens=int(usage["prompt_tokens"]) if usage.get("prompt_tokens") is not None else None,
            completion_tokens=int(usage["completion_tokens"]) if usage.get("completion_tokens") is not None else None,
        )


class UnslothBackend:
    def __init__(self, variant: Mapping[str, Any], root: Path) -> None:
        try:
            import torch
            from unsloth import FastVisionModel
        except ImportError as exc:
            raise PipelineError("Unsloth benchmark backend must run in compose.train.yaml") from exc
        if not torch.cuda.is_available():
            raise PipelineError("CUDA is required for the Unsloth benchmark backend")
        model_path = root / str(variant["model_path"])
        if not model_path.is_dir():
            raise PipelineError(f"Benchmark model path does not exist: {model_path}")
        self.torch = torch
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=str(model_path),
            load_in_4bit=bool(variant.get("load_in_4bit", False)),
            dtype=torch.bfloat16,
        )
        FastVisionModel.for_inference(self.model)

    def generate(self, messages: list[dict[str, str]], generation: Mapping[str, Any]) -> Generation:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(generation.get("enable_thinking", False)),
        )
        inputs = self.tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
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
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion = output[0, prompt_tokens:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return Generation(
            text=text,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion.shape[-1]),
        )


def build_backend(variant: Mapping[str, Any], root: Path) -> Any:
    backend = str(variant["backend"])
    if backend == "unsloth":
        return UnslothBackend(variant, root)
    if backend == "openai_compatible":
        return OpenAICompatibleBackend(variant)
    raise PipelineError(f"Unsupported benchmark backend: {backend}")
