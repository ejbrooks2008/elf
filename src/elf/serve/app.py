"""Serve models via Gradio or FastAPI (placeholder)."""
from __future__ import annotations

from pathlib import Path


def serve(config_path: Path | None = None, host: str = "0.0.0.0", port: int = 8000) -> int:
    print(f"[serve] host={host} port={port} config={config_path}")
    print("TODO: launch Gradio/FastAPI app and load adapters")
    return 0
