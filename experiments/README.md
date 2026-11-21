# Experiment record

Read [the plan](PLAN.md) and [decision log](../docs/DECISIONS.md) before adding
runs. `results.csv` indexes measured runs; compact metrics and curves live under
`metrics/`. Planned experiments belong in PLAN.md, not as invented scores in the
results file. See `notes/E001-seed42-v1.md` for the completed first baseline.

## Before a run

Assign a unique run ID (for example `E001-seed42-v1`). Record the hypothesis,
reference run, changed factors, owner, and decision ID in a short run note under
`experiments/notes/`. Save the exact configuration used, including any command-line
overrides. Record the code commit and whether the working tree was dirty.
For consequential choices, update `docs/DECISIONS.md` before implementing them.

## After a run

Append one row to `results.csv`; never overwrite an earlier run. Use accuracy as
a fraction in [0,1]. Leave unavailable fields empty rather than guessing.

| Field | Meaning |
| --- | --- |
| `run_id`, `date`, `owner` | Unique identity, execution date, responsible member |
| `decision_id`, `hypothesis`, `reference_run` | Why we ran it and what it compares with |
| `status` | `completed` or `failed`; explain failures in notes |
| `code_commit`, `dirty_worktree` | Code provenance; record uncommitted changes |
| `config_path`, `split_metadata_path` | Exact run settings and split identity |
| `seed`, `architecture`, `pretrained_source` | Training seed, model, exact weight identifier/source |
| `val_accuracy`, `val_examples`, `train_minutes` | Measured score, denominator, and runtime |
| `artifacts_path` | Local or team-accessible directory with the artifacts below |
| `notes_path` | Interpretation, tradeoffs, failures, and next action |

Keep the following with each run's artifacts:

- Resolved config, package versions, device, seeds, and checkpoint sources/hashes.
- Per-epoch training/validation curves and selected checkpoint/epoch.
- Validation IDs, true labels, predictions, probabilities, and explicit class order.
- Confusion matrix and class counts/accuracy; labels from the provided training CSV only.
- Checkpoint and SHA-256 checksum. Large outputs remain outside Git under ignored
  `outputs/` or `checkpoints/`; record an accessible location when sharing or submitting.

Keep compact configs, metrics, and explanatory notes in Git. Do not commit
images or credentials. Model output can contain sensitive team work: share only
within the assigned team and with instructors as required by the competition.

## Explain the result

In the run note, answer: What changed? Why might it help? What happened? What
else could explain the result? What will we do next? Distinguish a hypothesis
from a measured effect. Record regressions as well as improvements.

For important revisions, append a dated update to the relevant decision entry.
Record each member's actual work and how AI assisted; do not infer contributions
from who ran a command or retrospectively invent ownership.
