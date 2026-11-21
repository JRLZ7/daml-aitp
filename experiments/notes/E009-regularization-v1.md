# E009 — L2 regularization on the frozen E001 feature cache

Planned on 2026-11-21. Initial status: planned. Completed on 2026-11-21; results below.

## Question and ownership

Does a small L2 regularization sweep improve frozen-feature validation accuracy?
Jason requested experiments. Codex implements and executes this local suite;
other team contributions are not inferred. Decisions: D011 and D006. This note
covers all four predeclared candidate run IDs below.

## Fixed inputs and controlled changes

- Reference: E001-seed42-v1 (829/1,066 correct, 77.7674%).
- Use its exact cached train/validation features, class order, and split. Verify
  every ID and target against the labeled manifest, and hash source artifacts.
- No new images, image views, scaling, feature normalization, encoder fitting,
  class weights, augmentation, or test predictions.
- Fit four independent multinomial logistic regression classifiers with L2,
  lbfgs, `C = [0.01, 0.1, 1.0, 10.0]`, tolerance 1e-4, max iterations 1000.
  Smaller C means stronger regularization. No warm starts between candidates.
- Seed 42, CPU with four BLAS threads. Fit uses the 4,261 fitting rows only.
- Candidate run IDs: E009-regularization-v1-C0p01, E009-regularization-v1-C0p1,
  E009-regularization-v1-C1p0, E009-regularization-v1-C10p0.
- Record convergence, fitting/validation accuracy, cross-entropy, runtime,
  corrected/regressed examples, and descriptive paired bootstrap intervals.
- Export coefficients into an ordinary PyTorch scene checkpoint. Verify sklearn
  and exported predictions agree before recording the score; then re-evaluate
  the selected checkpoint using the actual validation images.

## Selection and limitations declared before running

Among converged candidates, select most validation correct; break ties by lower
validation log loss and then smaller C. Do not extend the grid after seeing scores.
If no candidate converges, record that outcome and make a new plan.

Between the four candidates, C is the only intentionally changed fitting setting.
Relative to E001, both optimizer/convergence procedure and regularization change:
do not attribute a baseline improvement solely to L2 strength. Raw features are
left unchanged to avoid mixing in standardization. Save final exported probabilities
as the authoritative checkpoint result.

Bootstrap intervals are descriptive comparisons on the same validation examples;
they do not correct for trying four candidates or earlier checkpoint selection.
A selected gain is provisional and needs confirmation with frozen choices.

## Run

```bash
python -m src.regularization --config configs/regularization.yaml --owner "Codex (AI-assisted local execution)"
```

Requires the local E001 cache and checkpoint. Outputs go to
`outputs/E009-regularization-v1/`; each candidate appends its own register row.
The suite captures code provenance once at launch; rows written during the suite
do not retroactively make its launch code dirty. Preserve existing run IDs.

## Outcome

Executed from clean code commit `f177c08db193ecdb7ba33a91ab876b955f49c258`.
All four fits converged with no recorded warnings. Each took about 1.6–2.0 seconds
including export/artifact work; feature extraction was reused rather than timed again.

| Model / C | Fitting accuracy | Validation accuracy | Correct / 1,066 | Validation loss | Corrected / regressed vs E001 |
| --- | --- | --- | --- | --- | --- |
| E001 baseline | 99.79% | **77.77%** | **829** | 0.7972 | — |
| 0.01 (strongest L2 here) | 86.44% | 75.42% | 804 | 1.1778 | 41 / 66 |
| 0.1 | 98.83% | 77.58% | 827 | 0.8058 | 38 / 40 |
| 1.0 | 100.00% | 76.64% | 817 | 0.8009 | 36 / 48 |
| 10.0 (weakest L2 here) | 100.00% | 76.17% | 812 | 0.9573 | 41 / 58 |

**Decision: retain E001.** The predeclared within-suite selection is C=0.1;
none of these candidates exceeds the baseline on primary validation accuracy.
Do not promote the best member of a weaker suite automatically.

## What we learned

- Reducing the fitting/validation gap is not itself the objective. Strong L2
  reduced fitting accuracy substantially but also reduced validation accuracy.
- The C=0.1 difference from E001 is only two images. Its descriptive paired
  bootstrap interval spans approximately -1.88 to +1.41 percentage points.
  This does not establish that E001 is truly better on unseen data; there is
  no measured improvement that justifies replacing it.
- Head fitting/regularization on the same views did not improve this grid's
  validation accuracy or loss. This is not evidence that all regularization
  choices fail, or a causal diagnosis of the remaining errors.
- C=0.1 corrects 38 errors and introduces 40. Different predictions could be useful
  for a future predeclared ensemble experiment, but disagreement alone does not
  demonstrate an ensemble gain. We did not tune ensemble weights here.
- No feature scaling, context changes, fine-tuning, test inference, or Kaggle
  submissions occurred. No grid expansion was made after observing outcomes.

Next: review validation errors and test a scene-context change, keeping head
settings and the split controlled. Fine-tuning remains a separate comparison.
Avoid repeatedly refining this grid against the same small validation set.

## Verification and artifacts

Nine automated checks passed. Exported float32 heads matched sklearn predictions.
All four saved checkpoint hashes were checked, their cached-feature predictions
reproduced, and all encoders remained identical to E001. Re-evaluating C=0.1
from actual validation images on MPS reproduced all 1,066 predicted labels.
See the compact suite summary for its maximum probability difference and hashes.

Local full artifacts: `outputs/E009-regularization-v1/`. Each candidate has a
full `best.pt`, SHA-256 checksum, resolved config, per-image probabilities,
confusion matrix, per-class report, and metrics. The suite stores source hashes,
package versions, and code provenance. Weights remain ignored by Git and have
not been uploaded or shared.

Tracked compact results: `experiments/metrics/E009-regularization-v1/`. Each
candidate has its own row in `experiments/results.csv`; no previous row was
replaced. The original E001 checkpoint and split files remain unchanged.
