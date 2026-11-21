# MIT Indoor67 Scene Classification

Duke Applied Machine Learning AI Training Program — Fall 2026

Classify indoor scene images into 67 categories using the provided competition
training data and permitted ImageNet-pretrained image encoders.

## Local setup

Use Python 3.11. Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the exact versions resolved on the initial macOS development machine:

```bash
python -m pip install -r requirements-macos-py311.lock.txt
```

The lock file records that environment; other OS/GPU environments may require
their own compatible PyTorch installation. Record their versions separately.

## Download competition data

Follow [data/README.md](data/README.md). With your Kaggle competition slug:

```bash
python scripts/download_data.py daml-aitp-fall-26-challenge --login
```

Keep credentials outside source files. Data, virtual environments, checkpoints,
and temporary predictions are ignored by Git. Final selected prediction CSVs
can be tracked under `submissions/final/`.

## Project structure

- `data/`: Download instructions and ignored local competition data
- `scripts/`: Setup and data download utilities
- `notebooks/`: Exploration and experiments
- `src/`: Training and inference code (currently scaffolding)
- `experiments/`: Results and comparisons
- `checkpoints/`: Ignored model weights
- `submissions/`: Predictions; retain selected final CSVs in `final/`

## Reproducibility

Retain split assignments, seeds, configs, validation results, pretrained-weight
sources, and each member's contributions. The final package must include the
selected CSV, Kaggle submission details, and weights or an instructor-accessible
link with a SHA-256 checksum. Exclude competition images and credentials.

## AI assistance

OpenAI Codex assisted with repository ignore rules, environment setup, the
competition download utility, and setup documentation. Team members should
review and understand the code and update this disclosure as work continues.
