# Jason / Spencer integration review

Date: 2026-11-21. Integration branch: `codex/integrate-team-workflow`.
Sources: Jason `6edbfc2` and Spencer `859926a`; original history is preserved by a
real merge, rather than copying work without ancestry or squashing run commits.

## Kept and adapted

| Work | Source | Resolution |
| --- | --- | --- |
| Verified split/manifest loader and hashes | Jason | Keep canonical `src/dataset.py` and split files |
| ResNet-50, inference, regularization, tests, decision records | Jason | Retain existing code, results, and local checkpoints |
| Populated exploration notebook | Spencer | Keep; adapt paths, class mapping, batches, and split loading |
| Classical notebook and feature/SVM method | Spencer | Keep; factor fitting/features into `src/classical.py` and call from notebook |
| Original classical metrics/report/predictions | Spencer | Preserve unchanged under `experiments/metrics/spencer-original/` |
| SVM / dummy integration rerun | New AI-assisted integration | Record independently as E010; do not rewrite Spencer's scores |

Spencer's committed predictions have exactly the same 1,066 validation IDs and
ground-truth labels as the team's saved split. The current classes.csv order is
also the alphabetical order used by his original SVM. These facts were checked,
not assumed from the common seed.

## Not adopted

- A second independently generated split or `data/validation_split.csv`.
- Duplicate dataset APIs using `image_id`/`image_path`, alternate label mappings,
  or the `data/Images` helper path. The original notebooks bypassed parts of that
  helper; keeping both implementations would create divergent interfaces.
- Empty transfer-learning notebook or redundant directory placeholders.
- Embedded notebook outputs, image galleries, and stale machine-specific output.
  Both populated notebooks remain usable but outputs are generated locally.

The repository was found public with old notebook image outputs in existing
history. The user explicitly chose to leave visibility unchanged. This integration
clears the current notebook files; it does not rewrite history or claim that old
outputs have disappeared from past public commits. No visibility change is made.

## Reproducibility and attribution

Spencer authored the original exploration and classical method. Jason's branch
contains the shared workflow and AI-assisted neural baseline/regularization work.
Codex performed conflict resolution, interface adaptation, shared-module factoring,
and the integration checks. This does not assign unobserved contributions to any
other member. Keep both source histories and original metrics.

Read `docs/TEAM_WORKFLOW.md` for future branches and handoffs. Use independent
task branches and reviewed PRs; do not continue both historical branches as
competing implementations of the same shared loader.

## Validation

Completed on 2026-11-21:

- 12 unit/source checks pass, including shared-data isolation and notebook output checks.
- Both notebooks execute end to end: 15 code cells and 5 rendered charts in total.
- Fixed classical rerun reproduces Spencer's 1.5009% dummy and 8.3490% SVM scores.
  Every one of 1,066 SVM predictions matches after ID alignment.
- Locally saved classical checkpoints reproduce predictions after reload.
- Original CSV archive is byte-identical to Spencer's source; all shared split
  and neural code/config files remain unchanged from Jason's source.
- Source merge commit cd2ce31 has both 6edbfc2 and 859926a as parents.
- All tracked notebook outputs are empty; rendered copies remain ignored/local.

See `experiments/notes/E010-classical-integration-v1.md` and its compact metric
folder for actual run provenance. The 77.77% neural baseline remains the reference.
The integration does not introduce a new model-selection split or a Kaggle upload.
