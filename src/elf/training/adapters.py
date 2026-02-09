"""Adapter utilities for LoRA/DoRA with transformers and diffusers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def wrap_transformer_with_lora(model: Any, config: Dict[str, Any]) -> Any:
    """Attach a LoRA adapter to a transformer model.
    Expected keys in config: r, alpha, target_modules, dropout, bias
    """
    print("[adapters] wrap_transformer_with_lora called")
    return model


def wrap_transformer_with_dora(model: Any, config: Dict[str, Any]) -> Any:
    """Attach a DoRA adapter to a transformer model."""
    print("[adapters] wrap_transformer_with_dora called")
    return model


def wrap_unet_with_lora(unet: Any, config: Dict[str, Any]) -> Any:
    """Attach a LoRA adapter to a diffusion UNet."""
    print("[adapters] wrap_unet_with_lora called")
    return unet


def save_adapters(model: Any, output_path: Path) -> None:
    print(f"[adapters] saving adapters to {output_path}")


def load_adapters(model: Any, adapter_path: Path) -> Any:
    print(f"[adapters] loading adapters from {adapter_path}")
    return model
