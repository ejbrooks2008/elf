"""Labeling and curation launchers (FiftyOne / Label Studio)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def launch_labeling(backend: str = "fiftyone", config_path: Optional[Path] = None) -> int:
    print(f"[label] backend={backend} config={config_path}")
    print("TODO: integrate with FiftyOne session or Label Studio API")
    return 0
