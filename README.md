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
python -c "from pathlib import Path; import pandas as pd; p=Path('data/raw'); a=pd.read_csv(p/'train.csv'); b=pd.read_csv(p/'test.csv'); c=pd.read_csv(p/'classes.csv'); s=pd.read_csv(p/'sample_submission.csv'); assert len(a)==5327 and len(b)==1340 and len(c)==67; assert a['label'].nunique()==67; assert a['id'].is_unique and b['id'].is_unique and s['id'].is_unique; assert set(b['id'])==set(s['id']); missing=[v for df in (a,b) for v in df['file_path'] if not (p/'images'/v).is_file()]; assert not missing, f'Missing images: {missing[:5]}'; print('Data files, counts, IDs, and image paths OK')"
```

The expected counts are **5,327 training images, 1,340 test images, and 67 classes**.
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

**Current status:** environment setup and the download script are implemented.
The files in `src/` are empty scaffolding; training, evaluation, and prediction
commands will be documented as those components are built.

The next implementation steps are:

1. Validate data and save a fixed stratified train/validation split under `splits/`.
2. Build the image loader and a reproducible pretrained-model baseline.
3. Record validation results and compare at least two meaningful changes.
4. Generate and validate an `id,label` submission CSV.

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

## Project layout

| Location | Purpose |
| --- | --- |
| `config.yaml` | Starting settings; the future training code must load them |
| `data/` | Download instructions and ignored local competition files |
| `scripts/` | Data download and setup utilities |
| `notebooks/` | Exploration and experiment notebooks |
| `src/` | Training and inference code scaffolding |
| `experiments/` | Validation results and experiment comparisons |
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
competition download utility, and setup documentation. Team members should
review and understand the code and update this disclosure as work continues.
