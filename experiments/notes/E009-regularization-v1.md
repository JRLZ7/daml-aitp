# E009 — L2 regularization on the frozen E001 feature cache

Planned on 2026-11-21. Status: planned; no results measured yet.

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

Pending execution. Record negative results and failed convergence as well as gains.
