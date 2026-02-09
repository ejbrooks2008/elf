"""Checkpoint export/import helpers."""
from __future__ import annotations

from pathlib import Path


def export_artifacts(input_path: Path, output_path: Path, fmt: str = "safetensors") -> int:
    print(f"[export] input={input_path} output={output_path} format={fmt}")
    print("TODO: load checkpoint/adapters and export in requested format")
    return 0
