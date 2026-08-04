from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_vram_callback(metrics_path: Path) -> Any:
    import torch
    from transformers import TrainerCallback

    class VramCallback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not torch.cuda.is_available():
                return
            gib = 1024 ** 3
            payload = {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 4),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 4),
                "device": torch.cuda.get_device_name(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return VramCallback()
