# Elf Training Framework

Framework for training and evaluating image/text generation models with LoRA/DoRA. This repo focuses on dataset prep, labeling, training, and evaluation. Generated outputs will be consumed by a separate backend service.

## Requirements

- Python 3.11+ (CUDA-enabled environment recommended)
- Conda (recommended for GPU stack)

## Setup

### Conda (recommended)

```pwsh
conda env create -f environment.yml
conda activate elf
```

### Pip-only (CPU)

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

CLI commands (see `python run.py --help`):

- Prepare data: `python run.py prepare-data --config configs/train/example_lora.yaml`
- Manual image curation (FiftyOne): `python scripts/curate_with_fiftyone.py --root data/raw`
- Rename + captions (wd + Florence ensemble, uncensored): `python scripts/rename_and_caption.py --root data/raw`
- Launch labeling: `python run.py label --backend fiftyone`
- Train (LoRA/DoRA): `python run.py train --config configs/train/example_lora.yaml`
- Evaluate: `python run.py eval --config configs/eval/image.yaml`
- Serve: `python run.py serve --config configs/serve/gradio.yaml`
- Export adapters: `python run.py export --input models/sd15-lora --output exports/sd15-lora.safetensors`

## Test

```pwsh
pytest
```

## Debug in VS Code

Use the launch configurations in `.vscode/launch.json`:

- **Run elf (run.py)**
- **Run elf module**

## Project Layout

```
configs/           # Training, inference, and evaluation configs
data/raw/          # Raw datasets and downloads
data/processed/    # Cleaned and processed datasets
data/labels/       # Captions, tags, and annotations
models/            # Checkpoints, LoRA/DoRA adapters
notebooks/         # Exploration and labeling notebooks
scripts/           # One-off utilities and runners
src/elf/datasets/  # Dataset loading & preprocessing
src/elf/training/  # Training loops & adapters
src/elf/evaluation/# Metrics and eval utilities
src/elf/utils/     # Shared helpers
```

### Tagging strategy (uncensored, realism-biased)

- Ensemble: wd-v1-4-swinv2 classifier (NSFW-aware tags) + Florence-2-large-ft (community, uncensored) detailed captions.
- Post-processing: 35% down-weight on anime/cartoon/lineart prefixes, stricter cutoffs for anime noise, canonical `wood_elf` forced, soft vocab for PNW forest, magic, attire, and near-realistic digital art (Alita-like).
- Thresholds/caps: base cutoff 0.28, global cap 80 tags, per-category caps to keep captions concise; anime tags capped at 5.
- Outputs: comma-separated tags saved to `data/labels/{id}.txt`, plus manifest entries in `data/labels/manifest.jsonl` with attributes including Florence caption and rating.
- Caching/resume: tagging cache under `.cache/tagging`; reruns of `rename_and_caption.py` reuse cache and skip existing label/manifest entries unless `--no-skip-existing` is passed. Manifest writes are line-buffered so work is not lost on interruptions. Florence is optional and will fall back to wd-only tagging if it errors (status recorded in manifest attributes).

### Performance defaults (RTX 3060 Ti, 32GB RAM, i7-12700KF)

- wd tagger uses fp16 on CUDA and micro-batches internally sized for ~8GB VRAM.
- Florence generation runs fp16, greedy decode with max 192 tokens to keep latency reasonable per image.
- If a run is interrupted, rerun `python scripts/rename_and_caption.py --root data/raw` to continue; processed IDs are read from the manifest and cached tags are reused, so no work is lost.
