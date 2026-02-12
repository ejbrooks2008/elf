# Elf Training Framework

Framework for training and evaluating image/text generation models with LoRA/DoRA. This repo focuses on dataset prep, labeling, training, and evaluation. Generated outputs will be consumed by a separate backend service.

## Requirements

- Python 3.11+ (CUDA-enabled environment recommended)
- Conda (recommended for GPU stack)

## Setup

This project uses **two conda environments** to avoid dependency conflicts:

| Environment | Purpose | torch | transformers | Config |
|-------------|---------|-------|--------------|--------|
| `elf` (`.conda/`) | Tagging & captioning pipeline | 2.5.1+cu121 | 4.49.0 | `environment.yml` |
| `elf-train` | Flux LoRA training (ai-toolkit) | 2.6.0+cu124 | 4.57.3 | `environment-train.yml` |

### 1. Tagging environment (elf)

```pwsh
conda env create -f environment.yml
conda activate elf
```

### 2. Training environment (elf-train)

```pwsh
conda create -n elf-train python=3.11 -y
conda run -n elf-train pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n elf-train pip install -r ai-toolkit/requirements.txt
conda run -n elf-train pip install "git+https://github.com/huggingface/diffusers@8600b4c10d67b0ce200f664204358747bd53c775" --force-reinstall --no-deps
```

### Pip-only (CPU, tagging only)

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
data/training/     # Flux training dataset (image + .txt pairs)
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

### Flux LoRA Training (8 GB VRAM)

The training pipeline targets **Flux.1-Dev** with a LoRA adapter, optimised for an RTX 3060 Ti (8 GB VRAM). The uncensored base model used at inference time is [`enhanceaiteam/Flux-Uncensored-V2`](https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2).

#### Data preparation

Convert the tagging manifest into ai-toolkit's expected format (flat folder of images + `.txt` caption files):

```pwsh
python scripts/prepare_flux_dataset.py --trigger elf_character --convert-jpg --clean
```

Options:
- `--trigger TEXT` — prepend a trigger word to every caption (recommended: `elf_character`)
- `--style-tier on_target` — only include on-target images
- `--min-conformance 2.0` — filter by conformance score
- `--ratings general,sensitive` — filter by rating
- `--convert-jpg` — convert PNG→JPG (saves disk space)
- `--max-caption-len N` — truncate captions to N characters
- `--dry-run` — preview without writing

Output goes to `data/training/` (ignored by git).

#### Training

Training uses [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) with a config tailored for 8 GB VRAM.
**Must be run in the `elf-train` environment** (see Setup above).

```pwsh
# Clone ai-toolkit (one-time)
git clone --depth 1 https://github.com/ostris/ai-toolkit.git

# Accept the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev
# then log in:
conda run -n elf-train huggingface-cli login

# Run training
conda run -n elf-train python ai-toolkit/run.py configs/train/flux_lora_elf.yaml
```

Key 8 GB VRAM settings in `configs/train/flux_lora_elf.yaml`:
- `quantize: true` — 8-bit mixed-precision quantisation
- `low_vram: true` — aggressive VRAM offloading (in `model:` section)
- `gradient_checkpointing: true` — trade compute for memory
- `optimizer: adamw8bit` — 8-bit AdamW
- `batch_size: 1` with `gradient_accumulation_steps: 4`
- `resolution: [512, 768]` — capped to fit in 8 GB
- `cache_latents_to_disk: true` — offload VAE latents to disk
- `trigger_word: elf_character` — auto-prepended by ai-toolkit
- LoRA rank 16, alpha 16 (scaling factor 1.0)

Checkpoints save to `output/elf_flux_lora/` every 250 steps.
