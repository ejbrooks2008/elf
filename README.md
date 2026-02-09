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
- Rename + captions (stub tagger): `python scripts/rename_and_caption.py --root data/raw`
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
