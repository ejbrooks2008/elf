"""Launch a FiftyOne session to inspect duplicates, outliers, and embeddings in data/raw.

Features:
- Loads all image files under data/raw (recursively) into a temporary FiftyOne dataset.
- Computes SHA256 for exact-duplicate detection.
- Optionally computes perceptual hash (pHash) if `imagehash` is installed for near-duplicates.
- Flags basic resolution outliers (very small/very large area or extreme aspect ratio).
- Optionally computes CLIP image embeddings (torch + transformers) and a 2D embedding view so you can use the scatterplot in FiftyOne.
- Launches FiftyOne App so you can review and delete/curate.

Usage:
    python scripts/curate_with_fiftyone.py \
        --root data/raw \
        --dataset-name elf-curation \
        --embed  \
        --clip-model openai/clip-vit-base-patch32 \
        --batch-size 16

Notes:
- Deletions/removals are not performed automatically; use the FiftyOne UI to select and act.
- If you want near-duplicate detection, install `imagehash`: `python -m pip install imagehash`.
- Embeddings require torch+transformers and will use CUDA if available.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fiftyone as fo
from fiftyone import Sample
try:
    import fiftyone.brain as fob
except Exception:  # pragma: no cover - older versions
    fob = None
from PIL import Image

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - optional
    torch = None
    CLIPModel = None
    CLIPProcessor = None

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - optional
    NearestNeighbors = None

try:
    import imagehash  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    imagehash = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def iter_image_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_EXTS:
                paths.append(Path(dirpath) / fname)
    return sorted(paths)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path) -> Optional[str]:
    if imagehash is None:
        return None
    with Image.open(path) as img:
        return str(imagehash.phash(img))


def basic_resolution_stats(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return w, h


def flag_outliers(sizes: List[Tuple[int, int]], low_pct: float = 1.0, high_pct: float = 99.0, max_aspect: float = 3.0) -> List[bool]:
    areas = [w * h for w, h in sizes]
    sorted_areas = sorted(areas)
    if not sorted_areas:
        return []
    def pct(p: float) -> int:
        k = max(0, min(len(sorted_areas) - 1, int(len(sorted_areas) * p / 100)))
        return sorted_areas[k]
    low_cut, high_cut = pct(low_pct), pct(high_pct)
    flags: List[bool] = []
    for (w, h), area in zip(sizes, areas):
        aspect = max(w / h, h / w) if h != 0 and w != 0 else float("inf")
        outlier = area <= low_cut or area >= high_cut or aspect > max_aspect
        flags.append(outlier)
    return flags


def build_dataset(root: Path, dataset_name: str) -> Tuple[fo.Dataset, List[Path]]:
    paths = iter_image_paths(root)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")

    sha_buckets: Dict[str, List[int]] = {}
    phash_buckets: Dict[str, List[int]] = {}
    sizes: List[Tuple[int, int]] = []

    samples: List[Sample] = []
    for idx, path in enumerate(paths):
        sha = sha256_file(path)
        phash = compute_phash(path)
        w, h = basic_resolution_stats(path)
        sizes.append((w, h))

        sha_buckets.setdefault(sha, []).append(idx)
        if phash is not None:
            phash_buckets.setdefault(phash, []).append(idx)

        samples.append(
            Sample(
                filepath=str(path),
                metadata=fo.ImageMetadata(width=w, height=h),
                dup_sha256=sha,
                dup_phash=phash,
            )
        )

    res_outliers = flag_outliers(sizes)
    for sample, is_outlier in zip(samples, res_outliers):
        sample["is_resolution_outlier"] = is_outlier

    dataset = fo.Dataset(dataset_name, overwrite=True)
    dataset.add_samples(samples)

    # Mark duplicates
    def mark_dupes(bucket: Dict[str, List[int]], field: str) -> None:
        for _, idxs in bucket.items():
            if len(idxs) > 1:
                # Keep the first as canonical, others as dupes
                for i in idxs[1:]:
                    dataset[i][field] = True
        dataset.save()

    mark_dupes(sha_buckets, "is_exact_dupe")
    if imagehash is not None:
        mark_dupes(phash_buckets, "is_phash_dupe")

    return dataset, paths


def compute_clip_embeddings(paths: List[Path], model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 16) -> Optional[List[List[float]]]:
    """Compute CLIP embeddings for images.

    Returns a list of embedding vectors (as Python lists) aligned with `paths`, or None if torch/transformers are missing.
    """
    if torch is None or CLIPModel is None or CLIPProcessor is None:
        print("[embed] torch/transformers not available; skipping embeddings")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[embed] loading {model_name} on {device} (safetensors, dtype={dtype})")
    model = CLIPModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    embeddings: List[List[float]] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            elif hasattr(feats, "last_hidden_state"):
                # pool CLS token
                feats = feats.last_hidden_state[:, 0]
            feats = torch.nn.functional.normalize(feats, dim=-1)
            embeddings.extend(feats.cpu().tolist())
        # close PIL Images
        for img in images:
            img.close()
        print(f"[embed] processed {min(i + batch_size, len(paths))}/{len(paths)}")

    return embeddings


def compute_uniqueness(embeddings: List[List[float]], n_neighbors: int = 2) -> Optional[List[float]]:
    """Compute a simple uniqueness score per sample based on nearest-neighbor cosine distance.

    Uniqueness is defined as the distance to the closest other sample (higher = more unique).
    """
    if NearestNeighbors is None:
        print("[unique] sklearn not available; skipping uniqueness scores")
        return None

    import numpy as np

    X = np.array(embeddings, dtype="float32")
    if len(X) < 2:
        return [1.0] * len(X)

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X)), metric="cosine")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:,0] is self-distance (0). Take next-nearest.
    closest = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
    uniqueness = closest.tolist()
    return uniqueness


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate images with FiftyOne")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-name", type=str, default="elf-curation")
    parser.add_argument("--embed", action="store_true", help="Compute CLIP embeddings for scatterplot")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    dataset, paths = build_dataset(args.root, args.dataset_name)

    if args.embed:
        embeds = compute_clip_embeddings(paths, model_name=args.clip_model, batch_size=args.batch_size)
        if embeds is not None:
            print("[embed] computing visualization (UMAP)")
            viz_kwargs = dict(
                embeddings=embeds,
                brain_key="clip_embeddings",
                method="umap",
                embedding_field="clip_embedding",
            )
            if fob is not None and hasattr(fob, "compute_visualization"):
                fob.compute_visualization(dataset, **viz_kwargs)
            else:
                dataset.compute_visualization(**viz_kwargs)

            uniq = compute_uniqueness(embeds)
            if uniq is not None:
                dataset.set_values("uniqueness", uniq)
                print("[unique] stored per-sample uniqueness (cosine distance to nearest neighbor)")
        else:
            print("[embed] skipped (missing torch/transformers)")

    # Helpful views
    view = dataset.sort_by("is_exact_dupe", reverse=True)
    session = fo.launch_app(view=view, remote=False)
    if args.embed:
        print("[embed] Open the 'Embeddings' plot in the FiftyOne UI and select brain_key='clip_embeddings'")
        print("[unique] You can also filter/sort on field 'uniqueness' (higher = more unique vs nearest neighbor)")
    print("Session URL:", session.url)
    session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
