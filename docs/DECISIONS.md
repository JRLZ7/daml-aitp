# Decision log

Record important choices here when making them, rather than reconstructing the
story before presenting. Distinguish observations, hypotheses, and measured
results. These entries were drafted with Codex; team members should review them.

For each new decision record: date, status, problem, choice, alternatives,
reasoning, evidence/artifacts, tradeoffs, and what would make us reconsider.
Add an experiment ID when performance is involved. Keep earlier entries and
append a dated update when a choice changes.

## D001 — Shared Python environment

- Date: 2026-11-21. Status: implemented.
- Choice: Python 3.11; initial machine uses 3.11.9. Record exact installed packages
  in `requirements-macos-py311.lock.txt`.
- Why: a tested shared environment makes setup and reproducing issues easier.
  This is a coordination choice, not a claim that 3.11 trains better than 3.13.
- Alternative: separate Python versions for each teammate.
- Evidence: dependency checks and a 67-class ResNet-18 CPU forward pass passed
  during setup. This checked installation, not model accuracy. Apple MPS was
  detected outside the execution sandbox; GPU training has not yet been tested.
- Tradeoff: the Mac lock file is not a guarantee of portability to CUDA/Linux.
  Record the actual environment used for each training run.
- Revisit if: the selected training platform requires a different supported version.

## D002 — Keep data and large artifacts outside Git

- Date: 2026-11-21. Status: implemented.
- Choice: download course data independently into ignored `data/raw/`; ignore
  credentials, environments, weights, and temporary outputs. Track code, configs,
  split IDs, compact results, and selected final prediction CSVs.
- Why: avoid distributing restricted images or credentials and keep the repo usable.
- Alternative: store all artifacts inside the source repository.
- Evidence: `.gitignore`, `data/README.md`; ignore rules were checked against
  representative data, credentials, and retained result paths.
- Tradeoff: final weights need an instructor-accessible location and checksum.
- Revisit if: a team-only artifact store is needed; preserve course restrictions.

## D003 — One fixed stratified validation split

- Date: 2026-11-21. Status: implemented.
- Choice: sort labeled training records by ID and make an 80/20 stratified split
  with seed 42. Reuse saved IDs for the main experiment comparisons.
- Why: preserve class representation and compare models on the same images.
- Alternatives: different random splits per run; full cross-validation initially.
- Evidence: `splits/metadata.json`, `splits/class_counts.csv`, and
  `scripts/prepare_data.py`. There are 4,261 fitting and 1,066 validation images;
  every class has 15–17 validation images. A separate rerun reproduced the files.
- Tradeoff: a single small validation set is noisy and can be overfit through
  repeated selection. Row stratification does not establish that visually
  similar images are absent across the split.
- Revisit if: a training-only similarity audit reveals related examples across
  the split, or finalist comparisons are unstable. Document any replacement
  split and rerun all compared baselines on it.

## D004 — Validate inputs before fitting

- Date: 2026-11-21. Status: implemented.
- Choice: check manifest schemas, counts, labels, ID/path uniqueness, split
  separation, sample/test IDs, and image decoding; record manifest hashes.
- Why: data and path errors can invalidate experiments before modeling starts.
- Alternative: discover input problems during training.
- Evidence: all 6,667 provided images passed verification and RGB decoding.
- Limits: CSV hashes do not hash image bytes. Path checks and decoding are not a
  perceptual-duplicate audit. Test image decoding supplies no labels and does not
  fit a model. Do not inspect test images to manually assign labels.
- Revisit if: course files change or new input failures appear.

## D005 — Frozen features first, then fine-tuning

- Date: 2026-11-21. Status: proposed; no training results yet.
- Choice: begin with an explicitly identified ImageNet-only ResNet-50 checkpoint
  and a trained 67-class linear head. Then compare fine-tuning and ConvNeXt-Tiny.
- Why: separate pipeline correctness and pretrained feature quality from gains
  due to adapting the encoder or changing architecture.
- Alternatives: train from scratch; start immediately with a large ensemble.
- Evidence needed: baseline validation predictions, accuracy, runtime, and
  checkpoint provenance. `config.yaml` currently says ResNet-18 as scaffolding;
  it does not establish the final baseline architecture.
- Tradeoff: the first baseline may not maximize accuracy; it creates a reference.
  Verify the exact checkpoint training data before downloading/using weights.
- Revisit if: available compute makes ResNet-50 impractical or measured results
  favor another architecture. Architecture names alone do not establish compliance.

## D006 — Select by validation accuracy and retain individual predictions

- Date: 2026-11-21. Status: planned.
- Choice: primary metric is overall accuracy. Also save per-image probabilities,
  true labels, class order, and confusion/per-class summaries on validation only.
- Why: match scoring while identifying failure modes and comparing ensembles on
  identical examples. Small per-class samples need cautious interpretation.
- Alternatives: choose by training loss, macro accuracy, or public leaderboard.
- Initial loss proposal: ordinary cross-entropy without class reweighting; observed
  class counts range only from 76 to 83. Test changes rather than assuming them.
- Evidence needed: run artifacts linked from `experiments/results.csv`.
- Tradeoff: repeatedly tuning on one validation set can still overfit it. Confirm
  finalists using a planned second split or a small cross-validation comparison.
- Revisit if: substantial preprocessing failures or class imbalance are discovered.

## D007 — Test scene context before adding architectural complexity

- Date: 2026-11-21. Status: hypothesis to test.
- Choice: compare whole-room context, gentler crops, resolution, and eventually a
  whole-image plus local-view representation using permitted ImageNet features.
- Why: hypothesize that some classes need layout while others need object detail.
  Aggressive crops could remove useful evidence; this has not been measured here.
- Alternatives: default augmentations throughout; add larger models immediately.
- Evidence needed: controlled experiments and validation-only error analysis in
  `experiments/PLAN.md`. Multi-scale literature motivates a question, not a claim
  that this team will obtain the published gains.
- Tradeoff: extra views cost inference time and can overfit a small validation set.
- Revisit if: crops/resolution fail to improve repeatable validation performance.

## D008 — Prefer tested gains and complementary errors

- Date: 2026-11-21. Status: planned.
- Choice: allocate runs to explicit hypotheses; confirm promising changes;
  ensemble models only when saved validation predictions demonstrate a gain.
- Why: improve learning per experiment and avoid spending the deadline on an
  unmanageable search. A different architecture may still make the same errors.
- Alternatives: many untracked trials; select by tiny score changes or leaderboard.
- Evidence needed: paired predictions, number of corrected/regressed examples,
  run seeds, training time, and finalist confirmation.
- Tradeoff: fewer architectures are explored. No method guarantees competition rank.
- Revisit if: new evidence identifies a stronger failure mode worth investigating.

## D009 — E001 frozen-feature training recipe

- Date: 2026-11-21. Status: selected for implementation; E001-seed42-v1 pending.
- Choice: explicit `ResNet50_Weights.IMAGENET1K_V2` from ImageNet-1K. Use its
  deterministic resize-232/center-crop-224 normalization for fitting and validation.
  Extract frozen 2048-dimensional features once, keeping all encoder layers and
  BatchNorm in evaluation mode; fit a linear 67-class head on CPU.
- Why: a small, inexpensive run isolates how useful pretrained features are and
  provides cached features for later head-only comparisons. No random augmentation
  means one cached feature vector represents exactly one specified image view.
- Head recipe: 30 epochs, AdamW, learning rate 0.001, weight decay 0.0001,
  batch size 128, cross-entropy, seed 42. Select best validation accuracy and keep
  the earliest epoch on ties. These are initial settings, not tuned optima.
- Alternatives: image augmentation per epoch, fully fine-tuned encoder, sklearn
  logistic regression, or an expensive head hyperparameter search immediately.
- Tradeoff: deterministic center crops can discard room edges; caching prevents
  fresh augmentation, and a simple optimization recipe may leave head performance
  untapped. CPU head training and accelerator feature extraction can yield small
  numerical differences across hardware. Save actual versions/device and predictions.
- Evidence needed: unchanged encoder tensors, validation metrics and probabilities,
  learning curves, source checksum, and a checkpoint reload evaluation.
- Config: `configs/baseline.yaml` is authoritative for E001. Root `config.yaml`
  remains the split-preparation config; its old model/training fields are unused
  scaffolding, not baseline settings. Avoid regenerating split metadata to change
  model hyperparameters.
- Source: [official ResNet-50 weights and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- Revisit after: baseline errors/curves identify optimization, context, or encoder
  limitations. Record measured outcomes in `experiments/notes/E001-seed42-v1.md`.

## 2026-11-21 outcome update — D005, D006, D009 / E001-seed42-v1

- Status: implemented and measured; supersedes initial proposed/planned statuses
  for the frozen baseline and validation-artifact recording above.
- Selected frozen ResNet-50 baseline reached 829/1,066 (77.7674%) local validation
  accuracy at epoch 30 of the 30 predeclared epochs. Fitting accuracy was 99.7888%.
- Reloaded full checkpoint reproduced all validation predictions on MPS; 318
  encoder tensors remained identical to the official ImageNet source. Five
  pipeline checks passed. See the run note and compact metrics for checksums.
- Decision: retain this run as the E001 reference. Best epoch is at the training
  boundary, so do not claim convergence. No change to split, training duration,
  or preprocessing was made in response to intermediate results.
- Interpretation: the fitting/validation gap motivates testing generalization
  and input context; it does not establish which change will improve accuracy.
  Confusion pairs are small-sample observations, not explanations by themselves.
- Next evidence: validation-error review and separate fine-tuning/context
  comparisons. No test predictions or Kaggle uploads were made for this run.

## D010 — Remove unused scaffolding after the first working baseline

- Date: 2026-11-21. Status: implemented at user request.
- Choice: remove the three empty notebooks and redundant directory placeholders;
  keep the working Python package and `src/__init__.py`. Remove notebook-only
  direct dependencies from minimal requirements; preserve the original environment
  lock file and all historical run records. No installed packages are uninstalled.
- Simplify root `config.yaml` to the class count, seed, and validation fraction
  actually consumed by preparation. Training uses `configs/baseline.yaml`.
- Why: show only usable functionality and eliminate misleading ResNet-18 settings
  and notebook commands. Add notebooks later when they contain actual analysis.
- Provenance: keep the existing split metadata and checkpoint intact. Data
  preparation preserves creation-time metadata when the full recomputed report
  differs only in the hash of unused config text. All other differences and
  changed split-file content still trigger review rather than replacement.
- Alternative: leave empty future-work files or rewrite historical metadata.
- Validation: re-run preparation and check all saved split files stay byte-identical;
  run pipeline checks. The original baseline config remains in Git history.
- Revisit if: a working notebook or additional artifact storage is implemented.

## D011 — Four-candidate regularization suite on cached E001 features

- Date: 2026-11-21. Status: predeclared for execution; E009-regularization-v1.
- Choice: four independent multinomial L2 logistic-regression fits with
  C=[0.01, 0.1, 1.0, 10.0], lbfgs, tol=1e-4, max_iter=1000, seed 42.
  Fit only the fitting split; use unchanged raw E001 features and class order.
- Why: test the inexpensive head-fitting explanation for the gap before changing
  image preprocessing or the encoder. Smaller C increases L2 regularization.
- Alternative: an AdamW weight-decay grid, broader automated search, or immediate
  fine-tuning. C isolates regularization within this suite; comparing with E001
  also changes optimization/convergence and cannot isolate regularization alone.
- Evidence: verify cached ID/label alignment and source hashes, save every
  candidate's probabilities, convergence, losses, accuracy, timing, and paired
  corrections/regressions. Export full PyTorch checkpoints; re-evaluate the selected
  candidate from actual images with the existing evaluator.
- Selection: converged candidates only; highest validation accuracy, then lower
  log loss, then smaller C. Four candidates fixed before running; no grid expansion
  based on results. Pairwise bootstrap intervals are descriptive, not corrected
  for selection. Additional split confirmation remains necessary for finalists.
- Tradeoffs: raw feature scales affect regularization; no standardization is
  introduced here. The selected score reuses validation and may be optimistic.
- Config/note: `configs/regularization.yaml`,
  `experiments/notes/E009-regularization-v1.md`. No test inference or Kaggle uploads.
- Implementation dependency: declare threadpoolctl directly to bound BLAS threads;
  it is already present in the saved environment. No new package install is needed.
- Revisit: after all four outcomes, decide whether to retain a head candidate
  and move to validation-error review/context/fine-tuning experiments.

## 2026-11-21 outcome update — D011 / E009-regularization-v1

- Status: completed. All four predeclared L2 fits converged with no warnings.
  Validation accuracy for C=[0.01,0.1,1,10] was [75.42%,77.58%,76.64%,76.17%].
- Decision: keep E001's 77.77% as the current reference. C=0.1 is selected within
  this suite only; it corrected 38 baseline errors and regressed on 40 (-2 net).
- Interpretation: a smaller fitting/validation gap does not guarantee higher
  validation accuracy. This grid found no improvement; it does not rule out all
  regularization methods or establish an independent estimate of relative quality.
- Evidence: all four checkpoint hashes and cached predictions checked; encoders
  identical to E001. Full-image MPS reload of C=0.1 reproduced all validation
  labels. Nine pipeline checks pass. Run note/compact summary retain the details.
- Next: validation-error review followed by a controlled scene-context experiment;
  fine-tuning is still separate. No post-hoc grid expansion or ensemble tuning,
  test inference, or Kaggle submission occurred.
