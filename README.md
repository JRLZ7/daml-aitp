# MIT Indoor67 Scene Classification

Duke Applied Machine Learning AI Training Program — Fall 2026.
Predict one of 67 indoor scene categories for each competition test image.

**Deadline:** December 5, 2026, at 11:59 p.m. Eastern, for both Kaggle and Gradescope.

## 1. Open the repository

Clone this repository using its GitHub **Code** button if you do not already have
it. Open a terminal in the cloned `daml-aitp` folder. Run all commands below there.

Check your current branch and local changes:

```bash
git status
git branch --show-current
```

Jason works on `jason-zhou`. Other teammates should use their own branches.
Save or commit existing work before switching branches.

## 2. Set up Python once

Use **Python 3.11** for the shared project environment. The initial local setup
was verified with Python 3.11.9. You can keep other Python versions installed;
create this project's environment using 3.11 explicitly.

On macOS or Linux, after installing Python 3.11:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, with Python 3.11 and the Python launcher installed:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` for subsequent commands; activation is optional.

Check the installation:

```bash
python --version
python -m pip check
python -c "import torch, torchvision, pandas, sklearn, kagglehub; print('Imports OK'); print('PyTorch:', torch.__version__)"
```

`requirements.txt` lists the project dependencies. To reproduce the initial
macOS/Python 3.11 dependency versions, use the following instead of the regular
requirements install:

```bash
python -m pip install -r requirements-macos-py311.lock.txt
```

The lock file records the initial Mac environment. Other operating systems and
GPU setups may need compatible PyTorch packages and their own version records.

## 3. Download the Kaggle competition data

1. Sign in to the [course competition](https://www.kaggle.com/competitions/daml-aitp-fall-26-challenge).
2. Join your assigned team's Kaggle team and accept the competition rules.
3. Generate an API token in [Kaggle settings](https://www.kaggle.com/settings/api).
4. With the project environment active, run:

```bash
python scripts/download_data.py daml-aitp-fall-26-challenge --login
```

Enter your token at the private prompt. Keep it out of source code, notebooks,
Git commits, and chat. Each teammate downloads their own data copy.

The script downloads files into `data/raw/`. Inspect the result and extract any
remaining image archive so the folder structure is:

```text
data/raw/
  train.csv
  test.csv
  classes.csv
  sample_submission.csv
  images/
    train/
    test/
```

Use the competition files, rather than the public Indoor67 mirror. The course
files contain the exact split and identifiers required for this challenge.

If credentials are already configured, omit `--login`. For more details and
custom download locations, see [data/README.md](data/README.md).

## 4. Check the downloaded data

Run this after arranging the files above:

```bash
python scripts/prepare_data.py
```

This checks the CSV schemas, IDs, class labels, image paths, and whether every
image decodes. It then saves a stratified 80/20 split of the labeled training
records using seed 42 from `config.yaml`.

The expected data counts are **5,327 training images, 1,340 test images, and 67
classes**. The split contains **4,261 fitting images and 1,066 validation images**.
Use the committed assignments in [splits/README.md](splits/README.md) for all
baseline comparisons; do not regenerate a different split for each experiment.

An image's path is `data/raw/images/` plus its CSV `file_path` value.

## 5. Work on the project

In each new terminal session, open the repository and activate the environment:

```bash
source .venv/bin/activate
```

Windows PowerShell: `.\.venv\Scripts\Activate.ps1`.

To open the notebooks:

```bash
python -m jupyterlab
```

In VS Code, choose the interpreter inside this repository's `.venv` folder for
Python files and notebooks. Run `deactivate` when you are finished in a terminal.

## Train the frozen ResNet-50 baseline

Use the committed shared split. From the repository root:

```bash
python -m unittest discover -s tests -v
python -m src.train --config configs/baseline.yaml --run-id E001-seed42-v1
```

The first run downloads the official ImageNet-1K ResNet-50 V2 weights. It extracts
fixed features once, then trains a linear classifier for 30 epochs. The encoder
stays frozen. See the [recipe decision](docs/DECISIONS.md#d009--e001-frozen-feature-training-recipe)
and [run note](experiments/notes/E001-seed42-v1.md) for rationale and limitations.

`configs/baseline.yaml` controls model training. Root `config.yaml` is used for
split preparation; its original model/training fields are unused scaffolding.
The trainer records the resolved config, code commit, environment, checkpoint
source/checksums, learning curves, validation predictions, and per-class results
in ignored `outputs/E001-seed42-v1/`, and appends a row to `experiments/results.csv`.
The full model is saved as `outputs/E001-seed42-v1/best.pt`; no external pretrained
weights download is needed when reloading it for inference.

Run IDs and output directories cannot be reused. For a new experiment, prepare
`experiments/notes/YOUR_RUN_ID.md`, choose a new run ID, and record what changed.
Use `--device cpu`, `--device mps`, or `--device cuda` to override automatic device
selection. Changing hardware may cause numerical differences.

To re-evaluate the saved baseline on its recorded validation set:

```bash
python -m src.evaluate --checkpoint outputs/E001-seed42-v1/best.pt --output-dir outputs/E001-seed42-v1-recheck
```

The recheck directory must not already exist. Compare its `val_predictions.csv`
with the original run's predictions. Do not treat re-evaluating the same checkpoint
on the same split as a new model-selection experiment.

When ready to generate a competition CSV (this command does not upload it):

```bash
python -m src.predict --checkpoint outputs/E001-seed42-v1/best.pt --output submissions/E001-seed42-v1.csv
```

The predictor preserves sample-submission ID order and uses the class order saved
with the checkpoint. Test prediction code is available; real test inference is
separate from the first baseline training/validation run.

Next: inspect validation errors, compare fine-tuning and context/resolution
changes, and confirm finalists. Fine-tuning is not implemented in this trainer.

## 6. Save and share work with Git

Before working, fetch updates and inspect your branch:

```bash
git fetch origin
git status
```

If your branch tracks its remote branch and your working tree is clean, use
`git pull --ff-only` to receive updates on that branch. Bring changes from other
branches into yours through your team's agreed pull-request or merge workflow.

After making changes:

```bash
git diff
git status --short
git add path/to/changed-file
git diff --cached
git commit -m "Describe your change"
git push -u origin HEAD
```

Replace `path/to/changed-file` with the actual file; repeat for additional files.
Review staged changes before committing. Pushing shares your current branch;
use a pull request to propose merging it into the team's main branch.

Track code, dependency versions, configs, split IDs, concise experiment results,
and selected final CSVs under `submissions/final/`. `.gitignore` excludes local
data, credentials, environments, weights, and temporary outputs. Ignore rules
do not remove files that are already tracked.

## Decisions and experiment records

Read the [decision log](docs/DECISIONS.md) for what we chose, why, alternatives,
tradeoffs, and evidence. Important decisions must be documented as work proceeds.
Mark ideas as hypotheses until experiments support them.

The [experiment plan](experiments/PLAN.md) describes our questions and priorities.
Follow [the experiment recording guide](experiments/README.md) before running
models; save configs, individual validation predictions, timing, and outcomes.
`experiments/results.csv` is the index of measured runs, including failed runs.
See `experiments/results.csv` and run notes for measured results; planned runs
have no score until execution completes.

## Project layout

| Location | Purpose |
| --- | --- |
| `config.yaml` | Split-preparation settings; old model/training fields are unused |
| `configs/baseline.yaml` | Explicit frozen ResNet-50 training recipe |
| `data/` | Download instructions and ignored local competition files |
| `scripts/` | Data download, validation, and split utilities |
| `splits/` | Shared fitting/validation IDs, class counts, and dataset hashes |
| `notebooks/` | Exploration and experiment notebooks |
| `src/` | Image loader, frozen baseline trainer, evaluation, and prediction |
| `tests/` | Data isolation, mapping, and frozen-encoder checks |
| `docs/DECISIONS.md` | Important choices, rationale, evidence, and tradeoffs |
| `experiments/` | Experiment plan, run records, and validation comparisons |
| `checkpoints/` | Ignored trained weights |
| `submissions/` | Predictions; selected final CSVs go in `final/` |

## Common setup problems

- **Python 3.11 command not found:** install Python 3.11, restart the terminal,
  and repeat the version check. Installing it does not require removing 3.13.
- **Module not found:** activate `.venv` and install the requirements using
  `python -m pip`, so packages go into the selected environment.
- **Kaggle authentication or 403 error:** confirm competition access, accepted
  rules, and a valid token; rerun the download command with `--login`.
- **Missing image paths:** check for an unextracted archive or an extra nested
  directory. Keep `file_path` values unchanged and fix the folder layout.

## Final submission checklist

Submit both Kaggle predictions and a reproducible code package through
[Gradescope](https://www.gradescope.com/courses/1405465/assignments/9021572)
by **December 5, 2026, at 11:59 p.m. Eastern**.

- Include all training/inference code, dependency versions, seeds, and exact run instructions.
- Include the final prediction CSV, Kaggle submission ID or timestamp, team name, and members.
- Include final weights or an instructor-accessible link, with a SHA-256 checksum.
- Record every pretrained-weight and external-code source, validation results,
  experiment summary, AI assistance, and each member's contributions.
- For a repository submission, record the repository URL and exact commit hash.
- Exclude competition images and credentials. Confirm inference from the saved
  weights reproduces the selected prediction CSV.

Use only permitted data and checkpoints: ImageNet-pretrained encoders are allowed;
models trained or fine-tuned on Indoor67, Places, or SUN, external labeled scene
data, and recovering public-source test labels are prohibited. Select models
using local validation. The competition permits five submissions per team per
day and up to two selected final submissions.

## AI assistance

OpenAI Codex assisted with repository ignore rules, environment setup, the
competition download and validation utilities, shared split generation, and
setup documentation, decision records, experiment planning, and baseline
training/evaluation/inference code and checks. Team members should
review and understand the code and update this disclosure as work continues.
