# Project working conventions

- Read `docs/DECISIONS.md` and `experiments/README.md` before consequential changes.
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
