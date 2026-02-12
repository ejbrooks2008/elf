"""Dataset ingestion, cleaning, and splitting scaffolds.

For interactive duplicate/outlier inspection, use `scripts/curate_with_fiftyone.py`.
This module remains a placeholder for automated batch prep pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def prepare_data(config_path: Optional[Path] = None) -> int:
    """Placeholder for automated data preparation.

    Recommended pipeline (to be implemented):
    - Load config (YAML/JSON) defining sources (HF datasets, local dirs, S3), filters, splits
    - Run quality/NSFW/aesthetic scoring, deduplication, and tagging
    - Write manifests to data/processed and labels to data/labels
    """
    print(f"[prepare-data] using config: {config_path}")
    print("TODO: implement ingestion, filtering, dedup, tagging, and splitting")
    print("For manual inspection now, run: python scripts/curate_with_fiftyone.py --root data/raw")
    return 0
