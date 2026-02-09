"""Image tagging scaffolding.

This module provides a thin wrapper that can be swapped for your preferred
community tagger (e.g., wd-v1-4-swinv2, deepbooru, BLIP/Florence/Qwen2-VL).
It is intentionally minimal so we can plug in the right model later without
rewriting downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class TagResult:
    tags: List[str]
    attributes: Dict[str, str]


def tag_image(path: Path, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto") -> TagResult:
    """Placeholder tagger.

    Replace this with a real community tagger (wd-* or a vision-language model).
    Keep the signature stable so scripts stay compatible.
    """
    # Minimal stub: returns only a filename-based tag; extend with real model calls.
    with Image.open(path) as img:
        _ = img.size  # noqa: F841
    return TagResult(tags=[path.stem], attributes={})