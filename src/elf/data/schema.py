"""Shared data schemas for images, captions, and labels.

These lightweight Pydantic-style stubs define the expected fields for curated examples.
Replace with full pydantic models if/when validation is needed at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ImageRecord:
    image_path: Path
    caption: Optional[str] = None
    tags: Optional[List[str]] = None
    safety_labels: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class TextRecord:
    text: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None
