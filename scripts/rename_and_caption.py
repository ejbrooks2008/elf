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
import logging
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image
from tqdm import tqdm

from elf.data.tagging import tag_image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_image_paths(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS and p.is_file()])


def _human_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def image_metadata(path: Path) -> Dict[str, str]:
    width = height = "?"
    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception:
        pass
    try:
        size_str = _human_bytes(path.stat().st_size)
    except Exception:
        size_str = "?"
    return {"width": width, "height": height, "size": size_str}


def preprocess_image(src: Path, dest: Path, max_dim: int = 1536) -> Tuple[Image.Image, Dict[str, str | int]]:
    """Resize once, save to dest, and return the in-memory RGB image plus metadata."""

    dest.parent.mkdir(parents=True, exist_ok=True)

    size_str = "?"
    try:
        size_str = _human_bytes(src.stat().st_size)
    except Exception:
        pass

    processed: Image.Image | None = None
    meta: Dict[str, str] = {"width": "?", "height": "?", "size": size_str}

    try:
        with Image.open(src) as im:
            w, h = im.size
            meta.update({"width": w, "height": h})
            processed = im.convert("RGB")

        scale = min(1.0, max_dim / float(max(processed.size) or 1)) if processed else 1.0
        if processed and scale < 0.999:
            new_size = (max(1, int(processed.size[0] * scale)), max(1, int(processed.size[1] * scale)))
            lanczos = getattr(Image, "Resampling", Image).LANCZOS
            resized = processed.resize(new_size, resample=lanczos)
            processed.close()
            processed = resized

        save_kwargs = {}
        fmt = (dest.suffix or ".jpg").lstrip(".").upper()
        if fmt in {"JPG", "JPEG"}:
            save_kwargs.update(dict(quality=90, optimize=True))
            fmt = "JPEG"
        elif fmt == "WEBP":
            save_kwargs.update(dict(quality=90, method=4))
        elif fmt == "PNG":
            save_kwargs.update(dict(optimize=True, compress_level=6))

        if processed is None:
            raise RuntimeError("failed_to_process_image")

        processed.save(dest, format=fmt, **save_kwargs)
        return processed, meta
    except Exception:
        if processed is not None:
            try:
                processed.close()
            except Exception:
                pass
        shutil.copy2(src, dest)
        reopened = Image.open(dest).convert("RGB")
        return reopened, meta


def load_manifest_records(manifest_path: Path) -> Tuple["OrderedDict[str, Dict[str, object]]", int]:
    """Read manifest entries and return the latest record per id (deduped)."""

    records: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    if not manifest_path.exists():
        return records, 0

    deduped = 0
    with manifest_path.open("r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            img_id = rec.get("id")
            if not img_id:
                continue
            if img_id in records:
                deduped += 1
            records[img_id] = rec

    return records, deduped


def rename_and_caption(
    root: Path,
    out_images: Path,
    out_labels: Path,
    manifest_path: Path,
    skip_existing: bool = True,
    use_cache: bool = True,
) -> int:
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_records, deduped_on_load = load_manifest_records(manifest_path)
    processed_ids = set(manifest_records.keys())

    paths = iter_image_paths(root)
    total = len(paths)

    for idx, src in enumerate(tqdm(paths, total=total, desc="Captioning", unit="img", dynamic_ncols=True), start=1):
        img_id = sha256_file(src)
        dest = out_images / f"{img_id}{src.suffix.lower()}"
        txt_path = out_labels / f"{img_id}.txt"

        skip_hit = skip_existing and (img_id in processed_ids or txt_path.exists())

        if skip_hit and dest.exists():
            meta = image_metadata(dest)
            tqdm.write(f"[{idx}/{total}] {src.name} | {meta['width']}x{meta['height']} | {meta['size']}")
            logger.info(
                "Skipping processed %s (%s x %s, %s) [%d/%d]",
                src.name,
                meta["width"],
                meta["height"],
                meta["size"],
                idx,
                total,
            )
            continue

        try:
            processed_img, meta = preprocess_image(src, dest)
        except Exception as exc:
            logger.error("Failed to preprocess %s: %s", src, exc)
            continue

        tqdm.write(f"[{idx}/{total}] {src.name} | {meta['width']}x{meta['height']} | {meta['size']}")
        logger.info(
            "Processing %s (%s x %s, %s) [%d/%d]",
            src.name,
            meta["width"],
            meta["height"],
            meta["size"],
            idx,
            total,
        )

        if skip_hit:
            processed_img.close()
            continue

        tag_result = tag_image(dest, use_cache=use_cache, image_obj=processed_img)
        processed_img = None

        # Build training-optimized .txt: tags on line 1, caption on line 2
        tag_line = ", ".join(tag_result.tags)
        caption_line = tag_result.attributes.get("caption", "")
        if caption_line:
            txt_content = f"{tag_line}\n{caption_line}"
        else:
            txt_content = tag_line
        txt_path.write_text(txt_content, encoding="utf-8")

        record = {
            "id": img_id,
            "image": str(dest.as_posix()),
            "caption_file": str(txt_path.as_posix()),
            "tags": tag_result.tags,
            "attributes": tag_result.attributes,
        }
        manifest_records[img_id] = record

    # Rewrite manifest deduped (latest record wins)
    with manifest_path.open("w", encoding="utf-8") as mf:
        for rec in manifest_records.values():
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_written = len(manifest_records)
    print(
        f"Wrote manifest to {manifest_path} (deduped {deduped_on_load} on load, {total_written} unique entries)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rename images and create captions for training")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-images", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--out-labels", type=Path, default=Path("data/labels"))
    parser.add_argument("--manifest", type=Path, default=Path("data/labels/manifest.jsonl"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Bypass tag cache and regenerate")
    parser.add_argument("--cache", dest="use_cache", action="store_true", help="Use tag cache (default)")
    parser.set_defaults(use_cache=True)
    args = parser.parse_args()

    return rename_and_caption(
        args.root,
        args.out_images,
        args.out_labels,
        args.manifest,
        skip_existing=args.skip_existing,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())