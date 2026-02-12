#!/usr/bin/env python
"""Prepare dataset for Flux LoRA training with ai-toolkit.

Reads data/labels/manifest.jsonl and data/processed/images/ and produces
a flat folder of image + caption .txt pairs in data/training/.

ai-toolkit expects:
  - A folder of images (jpg/jpeg/png) and matching .txt files.
  - Each .txt file contains only the caption text for the paired image.
  - Image and .txt share the same stem (basename without extension).

This script:
  1. Reads the manifest to get image paths, captions, and metadata.
  2. Optionally filters by style_tier, rating, or conformance_score.
  3. Copies (or converts PNG→JPG) images to data/training/.
  4. Writes matching .txt caption files with optional trigger word prepended.
  5. Reports statistics and any skipped entries.

Usage:
  python scripts/prepare_flux_dataset.py [OPTIONS]

Options:
  --root DIR          Project root (default: auto-detect from script location)
  --output DIR        Output directory (default: data/training)
  --trigger TEXT      Trigger word to prepend to every caption (e.g. "elf_character")
  --min-conformance N Minimum conformance_score to include (default: 0.0)
  --style-tier TIER   Only include images with this style_tier (e.g. "on_target")
  --ratings LIST      Comma-separated ratings to include (default: all)
  --convert-jpg       Convert PNG images to JPG (saves disk, ai-toolkit friendly)
  --max-caption-len N Truncate captions to N characters (default: no limit)
  --dry-run           Show what would be done without writing files
  --clean             Remove existing files in output dir before writing
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from PIL import Image


def _project_root() -> Path:
    """Walk up from this script to find the project root (has src/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent


def _clean_caption(caption: str) -> str:
    """Light cleanup on caption text for training.

    - Collapse multiple commas/spaces.
    - Strip trailing comma/space.
    - Remove dangling sentence fragments ending with ', the,' or similar.
    """
    # Remove "likely of caucasian descent" leftovers (belt-and-suspenders)
    caption = re.sub(
        r",?\s*(?:likely\s+)?(?:is\s+)?of\s+caucasian\s+descent\b",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace
    caption = re.sub(r"\s{2,}", " ", caption)
    # Collapse multiple commas
    caption = re.sub(r"(,\s*){2,}", ", ", caption)
    # Strip trailing punctuation artifacts
    caption = caption.strip().rstrip(",").strip()
    return caption


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load all records from manifest.jsonl."""
    records = []
    with open(manifest_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping manifest line {i}: {e}", file=sys.stderr)
    return records


def _filter_records(
    records: list[dict],
    *,
    min_conformance: float = 0.0,
    style_tier: Optional[str] = None,
    allowed_ratings: Optional[set[str]] = None,
) -> tuple[list[dict], list[dict]]:
    """Filter records by conformance, style, and rating. Returns (kept, skipped)."""
    kept, skipped = [], []
    for rec in records:
        attrs = rec.get("attributes", {})

        # Conformance filter
        try:
            score = float(attrs.get("conformance_score", "0"))
        except (ValueError, TypeError):
            score = 0.0
        if score < min_conformance:
            skipped.append(rec)
            continue

        # Style tier filter
        if style_tier and attrs.get("style_tier", "") != style_tier:
            skipped.append(rec)
            continue

        # Rating filter
        rating = attrs.get("rating", "general")
        if allowed_ratings and rating not in allowed_ratings:
            skipped.append(rec)
            continue

        kept.append(rec)

    return kept, skipped


def _prepare_one(
    rec: dict,
    root: Path,
    output_dir: Path,
    *,
    trigger: str = "",
    convert_jpg: bool = False,
    max_caption_len: int = 0,
    dry_run: bool = False,
) -> Optional[str]:
    """Prepare a single image+caption pair. Returns error string or None."""
    img_rel = rec.get("image", "")
    img_path = root / img_rel
    if not img_path.exists():
        return f"Image not found: {img_path}"

    rec_id = rec.get("id", img_path.stem)
    attrs = rec.get("attributes", {})
    caption = attrs.get("caption", "")

    if not caption:
        return f"No caption for {rec_id}"

    caption = _clean_caption(caption)

    # Prepend trigger word
    if trigger:
        caption = f"{trigger}, {caption}"

    # Truncate if requested
    if max_caption_len > 0 and len(caption) > max_caption_len:
        caption = caption[:max_caption_len].rsplit(",", 1)[0].strip()

    # Determine output filenames
    if convert_jpg:
        out_img = output_dir / f"{rec_id}.jpg"
    else:
        out_img = output_dir / f"{rec_id}{img_path.suffix}"
    out_txt = output_dir / f"{rec_id}.txt"

    if dry_run:
        return None

    # Copy or convert image
    if convert_jpg and img_path.suffix.lower() != ".jpg":
        img = Image.open(img_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out_img, "JPEG", quality=95)
    else:
        shutil.copy2(img_path, out_img)

    # Write caption
    out_txt.write_text(caption, encoding="utf-8")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Flux LoRA training dataset from manifest."
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--trigger", type=str, default="", help="Trigger word to prepend"
    )
    parser.add_argument(
        "--min-conformance", type=float, default=0.0, help="Min conformance score"
    )
    parser.add_argument(
        "--style-tier", type=str, default=None, help="Filter by style_tier"
    )
    parser.add_argument(
        "--ratings",
        type=str,
        default=None,
        help="Comma-separated ratings to include",
    )
    parser.add_argument(
        "--convert-jpg", action="store_true", help="Convert PNG→JPG"
    )
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=0,
        help="Truncate captions to N chars",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove existing output files first"
    )
    args = parser.parse_args()

    root = args.root or _project_root()
    output_dir = args.output or (root / "data" / "training")
    manifest_path = root / "data" / "labels" / "manifest.jsonl"

    print(f"Project root : {root}")
    print(f"Manifest     : {manifest_path}")
    print(f"Output dir   : {output_dir}")
    if args.trigger:
        print(f"Trigger word : {args.trigger}")

    if not manifest_path.exists():
        print(f"✗ Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # Load manifest
    records = _load_manifest(manifest_path)
    print(f"Manifest records: {len(records)}")

    # Filter
    allowed_ratings = (
        set(args.ratings.split(",")) if args.ratings else None
    )
    kept, skipped = _filter_records(
        records,
        min_conformance=args.min_conformance,
        style_tier=args.style_tier,
        allowed_ratings=allowed_ratings,
    )
    print(f"After filtering : {len(kept)} kept, {len(skipped)} skipped")

    if not kept:
        print("✗ No records to process after filtering.", file=sys.stderr)
        return 1

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean and not args.dry_run:
        existing = list(output_dir.glob("*"))
        if existing:
            print(f"Cleaning {len(existing)} existing files in {output_dir}")
            for f in existing:
                f.unlink()

    # Process each record
    errors = []
    success = 0
    for rec in kept:
        err = _prepare_one(
            rec,
            root,
            output_dir,
            trigger=args.trigger,
            convert_jpg=args.convert_jpg,
            max_caption_len=args.max_caption_len,
            dry_run=args.dry_run,
        )
        if err:
            errors.append(err)
        else:
            success += 1

    # Report
    action = "Would prepare" if args.dry_run else "Prepared"
    print(f"\n{action} {success} image+caption pairs")
    if errors:
        print(f"  ⚠ {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # Quick stats on captions
    if not args.dry_run and success > 0:
        lengths = []
        for txt_file in output_dir.glob("*.txt"):
            lengths.append(len(txt_file.read_text(encoding="utf-8")))
        if lengths:
            avg = sum(lengths) / len(lengths)
            print(f"\nCaption stats: {len(lengths)} files, "
                  f"avg {avg:.0f} chars, "
                  f"min {min(lengths)}, max {max(lengths)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
