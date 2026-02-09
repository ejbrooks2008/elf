"""Evaluation entrypoints for image and text models."""
from __future__ import annotations

from pathlib import Path


def run_eval(config_path: Path) -> int:
    print(f"[eval] config={config_path}")
    print("TODO: load prompts/datasets, run inference, compute metrics (FID/CLIP/BLEU/etc.)")
    return 0
