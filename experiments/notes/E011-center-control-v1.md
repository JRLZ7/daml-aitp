# E011-center-control-v1

Date: 2026-11-21. Initial status: predeclared; completed outcome below. Decision D014 (D013 in the original branch-local execution record).
Jason requested the preprocessing comparison; Codex implements and executes it.
Reference: E001-seed42-v1 (829/1,066, 77.7674%).

Use configs/center-control.yaml and the existing shared split/class order. Only the
preprocessing policy differs between the paired E011 runs. Both use the exact
ImageNet ResNet-50 encoder, fresh linear-head initialization with seed 42,
30 epochs of AdamW at 0.001, weight decay 0.0001, head batch size 128. Encoder
and BatchNorm stay frozen. Keep earliest epoch on accuracy ties; no grid expansion.

Policy: pretrained_weights_transforms. Center control reproduces the original resize-shorter-edge-232,
center-crop-224 recipe. Letterbox resizes the longest edge to 224 using PIL bilinear,
rounds dimensions with Python round (minimum 1), and pads to 224 centered with
RGB [124,116,104], followed by the same ImageNet normalization. This approximate
mean-color padding uses no learned data statistics. Images can be upsampled.

The treatment preserves all image content but changes effective object scale and
adds borders. A gain/loss cannot distinguish those mechanisms independently.
No random augmentation, encoder fitting, head-setting changes, new split, or test
inference. Preprocessing is stored explicitly in new checkpoints; old checkpoints
retain the original transform. Do not reuse the E001 feature cache for letterbox.

Before running, inspect a fixed sample of existing validation errors. Save
original/crop/padded views locally, not in Git. Quantify crop area; visual samples
are exploratory observations, not causal explanations or a reason to tune padding.

Report accuracy, selected-epoch loss, fitting accuracy, paired corrections and
regressions vs E001, timing, probabilities, checkpoint/source hashes, and reload
verification. All encoder tensors must match the original ImageNet source.
The control must reproduce E001; otherwise investigate before interpreting treatment.

Command:

```bash
python -m src.train --config configs/center-control.yaml --run-id E011-center-control-v1 --owner "Codex (AI-assisted local execution)"
```

## Outcome

Completed under the predeclared settings; detailed paired outcome follows.

## Pre-run input review

A fixed sample of eight E001 validation mistakes was drawn with seed 42 before
training either E011 variant. Local side-by-side galleries are in
`outputs/preprocessing-review/`. Center crops retain 62.15%–93.22% of the resized
image area in these examples. Original/padded views retain side shelves in the
videostore example and more room edges in the deli/laboratory examples. The
waiting-room square image changes little, illustrating that cropping is not a
complete explanation for mistakes. Padding also visibly makes some objects
smaller. These are exploratory observations, not relabeling or causal evidence.
No policy or hyperparameter was adjusted in response to this sample.

## Completed comparison (2026-11-21)

Both runs used training code commit `a7399a5c9fad7e7ed52a440fab6f9f4f4b782f50`.
The control launched with a clean tree. Letterbox launched with a dirty flag
because only the preceding control's result-register row had been appended;
its recorded git diff contains that row, not any modeling-code changes.

| Input policy | Validation accuracy | Correct / 1,066 | Selected epoch | Validation loss | Fitting accuracy |
| --- | --- | --- | --- | --- | --- |
| E011 center control | **77.7674%** | **829** | 30 | **0.797219** | 99.7888% |
| E011 whole-image padding | 77.2045% | 823 | 29 | 0.816414 | 99.8592% |

**Decision: retain E001 center cropping as the reference.** Whole-image padding
corrected 57 baseline errors and regressed on 63, for six fewer correct images
(-0.5629 percentage points). Predictions disagreed on 178 images; that does not
by itself demonstrate an ensemble benefit. No ensemble was tuned or evaluated.

The descriptive paired bootstrap interval for the accuracy difference is about
-2.63 to +1.41 percentage points. It is not adjusted for checkpoint/model selection
and does not establish a reliable unseen-data advantage for either policy. The
observed result does not justify replacing the existing reference.

The center control exactly reproduced E001's full 30-epoch history, validation
predictions/probabilities, and all 320 model tensors. Its checkpoint-file checksum
differs because it now contains additional preprocessing/provenance metadata.
All 318 encoder tensors in both runs remained identical to E001/ImageNet.
The padded checkpoint reproduced all 1,066 labels when reloaded with its saved
policy and evaluated on actual images. See the compact summary for checksums and
maximum probability difference. Sixteen tests passed, including legacy checkpoint
compatibility, padding geometry, and saved-policy validation.

Each run took about 55–56 seconds, including fresh feature extraction. Full local
artifacts remain under their own `outputs/E011-.../` directories. Compact results,
histories, per-class counts, paired predictions, and verification are under
`experiments/metrics/E011-preprocessing/`. No images are committed.

This evaluates one 224-pixel padding policy. Retaining context also reduced object
scale and introduced borders, so it does not prove that context is irrelevant or
that all whole-image/multi-view methods fail. No settings were changed after
observing scores, no new fine-tuning was run, and no test predictions or Kaggle
submissions were made. Higher resolution/multi-view preprocessing and a separate
fine-tuning task remain future work.

## Synchronization with concurrent main changes

Main added Spencer's historical ResNet18 checkpoint/submission support during
this task. It is preserved, along with its archived results and notebook path
edits. Main's independently allocated D013 remains S001; E011 is now D014 in
shared documentation. Raw E011 records keep their original D013/source-commit
provenance and are not retroactively presented as a new run. Config metadata was
updated to the unambiguous decision ID; all modeling settings remain identical.

After the main synchronization, all **19** tests pass, including the three new
ResNet18 checkpoint regressions. Actual-image padded ResNet50 re-evaluation still
reproduces all 1,066 labels with the combined loader. No model was retrained.
