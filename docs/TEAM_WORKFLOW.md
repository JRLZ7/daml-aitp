# Shared workflow for teammates and Codex

Read this with `AGENTS.md`, `docs/DECISIONS.md`, and `experiments/README.md` before
changing the project. The README contains environment and execution commands.

## Branches and collaboration

- `main` is the reviewed working version. Work on a separate branch for one task.
- Start a new task from current `origin/main`, not an old personal branch. Use
  names such as `codex/spencer-validation-analysis` or `codex/jason-context-input`.
- Do not have two people or agents push to the same task branch independently.
  Coordinate ownership of changes to shared loaders, split logic, and registries.
- Different checkouts can use different branches. Within one checkout all
  editors/agents share the same files and current branch: do not switch branches
  or finish another active agent's merge while it is working.
- Use pull requests into `main`. Another teammate reviews the code and recorded
  evidence. Integrate current `main`, resolve conflicts deliberately, and test
  before merging. Keep merge commits when preserving recorded run commit hashes.
- Do not use blanket "ours/theirs" resolution for a whole merge, rewrite shared
  history, or overwrite another member's results. Preserve source attribution.

For a new task after this integration is merged:

```bash
git fetch origin
git switch main
git pull --ff-only
git switch -c codex/YOUR-NAME-TASK
```

Start with a clean working tree. Commit your own current work before switching;
never discard another agent's changes to make a branch switch succeed.

## One data contract

- Download the course competition to `data/raw/` on each machine.
- Manifest columns remain `id`, `file_path`, and training-only `label`.
- Image root is lowercase `data/raw/images/`; append the CSV `file_path` once.
- Read splits with `src.dataset.load_training_records(data_dir, split_dir)`.
  It returns fitting records, validation records, class names, and split metadata.
- Use the committed ID files under `splits/`. Do not call a new random split in
  an experiment or notebook. The class index order comes from `classes.csv`.
- Use `SceneDataset(records, image_root, classes, transform)` when loading images
  for PyTorch. Each item is `(image_tensor, target_index, id)`; unlabeled targets
  are `-1`. Do not reintroduce `image_id`/`image_path` aliases or a second loader.
- Root `config.yaml` controls data validation/split preparation. Experiment
  settings live in `configs/`. Do not change split metadata to change a model.

## Python modules and notebooks

Reusable features, fitting, and inference belong in `src/`. Notebooks import or
call them and explain results. This prevents notebook and CLI implementations
from silently diverging.

- `src.classical`: Spencer's RGB thumbnail/histogram + SVM method.
- `src.train`: frozen ResNet-50 baseline.
- `src.regularization`: predeclared cached-feature L2 suite.
- `src.evaluate` and `src.predict`: saved neural checkpoint evaluation/inference.
- `notebooks/01_data_exploration.ipynb`: shared-data inspection and batch example.
- `notebooks/02_baseline.ipynb`: classical results, confusion, and validation errors.

Notebook 02 loads existing local artifacts or tracked compact results by default.
It trains only when explicitly requested with a new run ID. Core commands require
`requirements.txt`; Jupyter tools are optional in `requirements-notebooks.txt`.

Keep notebook outputs empty in Git. Image galleries are generated only on local
course data. Before committing:

```bash
python scripts/clear_notebook_outputs.py notebooks/01_data_exploration.ipynb notebooks/02_baseline.ipynb
```

## Experiments and agent handoff

1. Agree on a unique run ID, including an owner/task suffix when useful. Check
   `experiments/results.csv` and pending run notes so parallel work does not collide.
2. Add a pre-run note and config. State the hypothesis, reference, changed factors,
   owner, data/split, and selection rule. Document important choices in DECISIONS.
3. Commit code/config before running where practical. Record actual code commit,
   dirty status, package versions, seeds, checkpoint sources, and execution owner.
4. Fit only on fitting rows; evaluate on saved validation IDs. Do not learn feature
   scaling or normalization statistics from validation/test rows. Never recover
   test labels. Use only course-permitted checkpoints/data.
5. Append new result rows; retain failures, convergence warnings, and negative
   results. Never replace historical rows to make a rerun look identical.
6. Keep weights/caches/image galleries under ignored local output directories.
   Commit small summaries, configs, predictions/reports where useful, and notes.
7. Explain measured outcomes, limitations, and actual member/AI contributions.
   A smaller train/validation gap alone is not the optimization objective.

Spencer's original results are archived under `experiments/metrics/spencer-original/`.
New integration reruns have their own IDs. Matching validation membership does
not mean different preprocessing, optimizers, and checkpoints are interchangeable.

## Checks before a pull request

```bash
python -m unittest discover -s tests -v
git diff --check
git status --short
git diff --cached
```

Run affected CLI commands and changed notebooks end to end on local course data.
Clear notebook outputs afterwards. State which checks actually ran and which
assets are local-only. Do not claim an unexecuted path works.

## Prompt to give another Codex session

> Read AGENTS.md, docs/TEAM_WORKFLOW.md, docs/DECISIONS.md, and experiments/README.md.
> Inspect git status and the current branch before editing. Start this task from
> updated main on a separate task branch. Reuse the shared loader, classes.csv
> order, data/raw layout, and committed split IDs. Keep reusable code in src and
> notebook outputs empty. Document important choices and each experiment with
> actual provenance and contributions. Preserve existing results and other
> teammates' work; test the affected paths before proposing a merge.
