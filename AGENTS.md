# Project working conventions

- Read `docs/TEAM_WORKFLOW.md`, `docs/DECISIONS.md`, and `experiments/README.md` before consequential changes.
- Document important data, validation, architecture, preprocessing, training,
  inference, and reproducibility decisions in `docs/DECISIONS.md` with the change.
  Include the reason, alternatives/tradeoffs, evidence, and revisit conditions.
  Mark untested choices as proposed/planned; never imply measured gains without runs.
- Link modeling decisions to experiment IDs. Record completed/failed runs and
  actual member/AI contributions according to `experiments/README.md`.
- Update runnable instructions when behavior changes. Do not present empty
  scaffolding or unexecuted commands as working training/inference functionality.
- Reuse the committed shared split for main comparisons. Document any deliberate
  split replacement and rerun comparable baselines; keep test data out of fitting.
- Use only course-permitted data and checkpoints. Check exact pretrained-weight
  provenance; an architecture name does not establish compliance. Never recover
  test labels or redistribute competition images or credentials.
- Keep work scoped to the current user request and preserve other members' work.

## Shared interface and handoff

- `main` is the reviewed integration point. Use an independent task branch from
  current main; do not have multiple agents independently edit the same checkout
  or complete another active agent's merge. Preserve source commits and credit.
- The only shared loader is `src.dataset`: `load_training_records` returns fitting
  records, validation records, class names, metadata. `SceneDataset` returns
  `(image_tensor, target_index, id)`. Use `id/file_path/label`, `data/raw/images/`,
  committed split IDs, and classes.csv order; do not regenerate a local split.
- Reusable code belongs in `src/`; notebooks call it and explain results. Use
  `requirements-notebooks.txt` for optional Jupyter tools. Commit notebook source
  with empty outputs; generated dataset galleries remain local. Use the provided
  output-clearing script and run the notebook-source checks before committing.
- Reserve unique run IDs and pre-run notes before parallel experiments. Append
  outcomes, including failures; do not overwrite other members' rows or artifacts.
- Read `docs/INTEGRATION.md` for what was kept from Jason and Spencer. Preserve the
  original classical CSVs and source attribution; an integration rerun is a new
  result, not a correction of historical scores. All claims must link actual runs.
