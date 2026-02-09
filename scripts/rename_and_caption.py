"""Rename curated images and emit captions/metadata for training.

Workflow:
- Reads images under data/raw (already curated).
- Computes a stable SHA256-based id per image.
- Copies images to data/processed/images/{id}.ext.
- Generates a text caption file data/labels/{id}.txt with placeholder tags.
- Writes a manifest data/labels/manifest.jsonl with paths and tags.

Hook points:
- Replace `tag_image` in elf.data.tagging with a real community tagger
  that supports NSFW and rich attributes (wd-swinv2, Qwen2-VL, Florence, etc.).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import List

from elf.data.tagging import tag_image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_image_paths(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()])


def rename_and_caption(root: Path, out_images: Path, out_labels: Path, manifest_path: Path) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        for src in iter_image_paths(root):
            img_id = sha256_file(src)
            dest = out_images / f"{img_id}{src.suffix.lower()}"
            if not dest.exists():
                shutil.copy2(src, dest)

            tag_result = tag_image(src)
            caption = ", ".join(tag_result.tags)
            txt_path = out_labels / f"{img_id}.txt"
            txt_path.write_text(caption, encoding="utf-8")

            record = {
                "id": img_id,
                "image": str(dest.as_posix()),
                "caption_file": str(txt_path.as_posix()),
                "tags": tag_result.tags,
                "attributes": tag_result.attributes,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote manifest to {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename images and create captions for training")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-images", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--out-labels", type=Path, default=Path("data/labels"))
    parser.add_argument("--manifest", type=Path, default=Path("data/labels/manifest.jsonl"))
    args = parser.parse_args()

    return rename_and_caption(args.root, args.out_images, args.out_labels, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())