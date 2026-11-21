# E011-center-control-v1

Date: 2026-11-21. Status: predeclared, not yet executed. Decision D013.
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

Pending execution. Record negative outcomes; do not extend training after scores.

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
