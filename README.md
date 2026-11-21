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

`main` is the shared reviewed version. Start new tasks on separate branches
from updated main; the historical `jason-zhou` and `skolodny/initialTesting`
branches are preserved as sources of this integration. Save your own work before
switching branches and do not switch a checkout another agent is actively using.

**For teammates and Codex:** read [AGENTS.md](AGENTS.md), the
[shared workflow](docs/TEAM_WORKFLOW.md), [decision log](docs/DECISIONS.md), and
[experiment recording guide](experiments/README.md) before editing. The shared
workflow includes a ready-to-copy Codex handoff prompt. The
[integration review](docs/INTEGRATION.md) explains what was kept from each branch.

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

The lock file records the original baseline Mac environment, including notebook
tools removed from the current minimal requirements. It is retained for provenance. Other operating systems and
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

Reusable training, features, and inference use `.py` modules. Populated notebooks
are retained for exploration and explaining results; they call the same shared
code. In VS Code, select the interpreter inside `.venv`. Run `deactivate` when
you are finished in a terminal.

For optional notebooks, install and launch:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab
```

Open `notebooks/01_data_exploration.ipynb` for data inspection or
`notebooks/02_baseline.ipynb` for Spencer's classical baseline and error analysis.
Both use `data/raw/`, `classes.csv`, and the committed shared split. Notebook 02
loads recorded results by default; it does not silently retrain or overwrite runs.
Run all cells on your local course data, then clear outputs before committing:

```bash
python scripts/clear_notebook_outputs.py notebooks/01_data_exploration.ipynb notebooks/02_baseline.ipynb
```

## Run the classical baseline

Spencer's RGB-thumbnail/histogram + SVM method is shared by the CLI and notebook:

```bash
python -m src.classical --config configs/classical.yaml --run-id E010-classical-integration-v1 --owner "YOUR NAME"
```

The documented integration run ID may already be recorded. For a new rerun, first
choose a fresh ID, copy the pre-run note to `experiments/notes/YOUR_RUN_ID.md`,
update its hypothesis/ownership, and replace the command's run ID. Existing IDs
and output directories are never reused. Original Spencer metrics remain under
`experiments/metrics/spencer-original/`; reruns have independent provenance.

The fixed recipe uses an 816-dimensional RGB-thumbnail/histogram representation,
training-only StandardScaler, and LinearSVC(C=0.01). It also runs a most-frequent
reference. No ImageNet or external scene data is used for these classical fits.
**Verified integration rerun:** 1.50% most-frequent and 8.35% SVM accuracy;
all 1,066 SVM predictions match Spencer's original records.
See [the integration run note](experiments/notes/E010-classical-integration-v1.md).
Notebook 02 can display tracked compact results even without local checkpoints.
To explicitly train from the notebook, follow its fresh-ID/environment instructions;
normal notebook execution only reads the recorded run.

## Train the frozen ResNet-50 baseline

Use the committed shared split. From the repository root:

```bash
python -m unittest discover -s tests -v
python -m src.train --config configs/baseline.yaml --run-id E001-seed42-v1
```

**Measured E001 baseline:** 77.77% local validation accuracy (829/1,066),
selected at epoch 30. All validation predictions reproduced after checkpoint reload.
See [the completed run note](experiments/notes/E001-seed42-v1.md) for details.

The first run downloads the official ImageNet-1K ResNet-50 V2 weights. It extracts
fixed features once, then trains a linear classifier for 30 epochs. The encoder
stays frozen. See the [recipe decision](docs/DECISIONS.md#d009--e001-frozen-feature-training-recipe)
and [run note](experiments/notes/E001-seed42-v1.md) for rationale and limitations.

`configs/baseline.yaml` controls model training. Root `config.yaml` is used for
split preparation only.
The trainer records the resolved config, code commit, environment, checkpoint
source/checksums, learning curves, validation predictions, and per-class results
in ignored `outputs/E001-seed42-v1/`, and appends a row to `experiments/results.csv`.
The full model is saved as `outputs/E001-seed42-v1/best.pt`; no external pretrained
weights download is needed when reloading it for inference.

Run IDs and output directories cannot be reused. For a new experiment, prepare
`experiments/notes/YOUR_RUN_ID.md`, choose a new run ID, and record what changed.
For new runs, add `--owner "YOUR NAME"` to record who executed the run; this
is separate from contribution credit in run notes. Use `--device cpu`, `--device mps`, or `--device cuda` to override automatic device
selection. Changing hardware may cause numerical differences.

To re-evaluate the saved baseline on its recorded validation set:

```bash
python -m src.evaluate --checkpoint outputs/E001-seed42-v1/best.pt --output-dir outputs/E001-seed42-v1-recheck
```

The example run and recheck already exist on Jason's machine. Use a new run ID
and matching pre-run note to train another model, or a new output directory to
re-evaluate. The recheck directory must not already exist. Compare its `val_predictions.csv`
with the original run's predictions. To verify labels, probabilities, hashes,
and unchanged encoder tensors automatically (requires the original downloaded
ImageNet checkpoint in the local torch cache):

```bash
python scripts/verify_baseline.py --run-dir outputs/E001-seed42-v1 --recheck-dir outputs/E001-seed42-v1-recheck
```

Do not modify run artifacts to make a verification pass. Do not treat re-evaluating the same checkpoint
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

## Run the regularization experiment

After the E001 feature cache and checkpoint exist locally:

```bash
python -m src.regularization --config configs/regularization.yaml --owner "YOUR NAME"
```

This fits four L2 linear classifiers on the same frozen features and saves each
as a full scene checkpoint under `outputs/E009-regularization-v1/`. It compares
validation accuracy, log loss, and corrected/regressed examples with E001.
The test set is not used. The [suite note](experiments/notes/E009-regularization-v1.md)
records the hypothesis, settings, and limitations.

**Measured E009 result:** all four fits converged; none exceeded E001. Best
within the suite was C=0.1 at 77.58%, so the 77.77% baseline remains our reference.
See [the comparison](experiments/metrics/E009-regularization-v1/comparison.csv).

Suite IDs cannot be reused;
copy the config and prepare a new note/ID for an intentional repeat.

## 6. Save and share work with Git

Use a fresh task branch from updated main for new work:

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c codex/YOUR-NAME-TASK
```

Check that your working tree is clean before switching. Do not work concurrently
on the same task branch or checkout. Keep source history intact; use reviewed
pull requests into main rather than directly replacing files from another branch.
See [the full team workflow](docs/TEAM_WORKFLOW.md).

While working, fetch updates and inspect your branch:

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
| `config.yaml` | Data validation and shared split settings |
| `configs/` | Fixed classical, ResNet-50, and regularization recipes |
| `data/` | Download instructions and ignored local competition files |
| `scripts/` | Data download, validation, and split utilities |
| `splits/` | Shared fitting/validation IDs, class counts, and dataset hashes |
| `src/` | Shared loader, classical/ResNet baselines, regularization, evaluation, prediction |
| `notebooks/` | Populated exploration and classical analysis; no committed outputs |
| `tests/` | Data isolation, mapping, and frozen-encoder checks |
| `docs/DECISIONS.md` | Important choices, rationale, evidence, and tradeoffs |
| `experiments/` | Experiment plan, run records, and validation comparisons |
| `outputs/` | Generated run artifacts, including model weights; ignored by Git |
| `submissions/` | Created when predicting; selected final CSVs go in `final/` |

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
training/evaluation/inference code and checks, regularization experiments, and
branch integration. Spencer authored the original exploration and classical
feature/SVM method; Codex adapted it to the shared loader and reusable module.
See the integration review and run notes for actual execution/contribution records. Team members should
review and understand the code and update this disclosure as work continues.
