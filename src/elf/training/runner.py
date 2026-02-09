"""Training entrypoints for LoRA/DoRA runs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def train(config_path: Path, resume: bool = False) -> int:
    print(f"[train] config={config_path} resume={resume}")
    print("TODO: load config, build model/dataset, apply LoRA/DoRA, run training with Accelerate")
    return 0
