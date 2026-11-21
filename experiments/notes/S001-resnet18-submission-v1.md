# S001 — Kaggle submission preparation

Date: 2026-11-21. Status: completed. Decision: D013.

Spencer requested a Kaggle-ready submission from the best trained model.
Codex implements historical-checkpoint compatibility, runs inference and checks
the resulting CSV. The team must review and understand submitted code; this
does not assign unobserved work to other members.

## Fixed model and inference recipe

- Source: `checkpoints/resnet18_low_lr_best.pt`, the best checkpoint available
  locally, selected at epoch 5 with 68.1989% validation accuracy and 0.6801 macro F1.
- Recorded reference: `experiments/metrics/spencer-original/resnet18_low_lr_metrics.csv`.
  The team's 77.7674% E001 ResNet50 checkpoint is not present locally. This run
  must not be described as using that model or achieving its score.
- Exact pretrained source: `ResNet18_Weights.IMAGENET1K_V1`; restore the complete
  trained state without any pretrained download or further training.
- RGB, resize shorter edge to 256, center crop 224, ImageNet normalization.
- Use the shared `SceneDataset` and explicit saved class order. Local manifest
  root is `data/`, image root `data/images/`; shared CLI defaults stay unchanged.
- Predict test labels from model outputs only, matching sample IDs/order/schema.
  No test inspection, label recovery, test-time tuning, or Kaggle upload.
- Output: `submissions/final/resnet18_low_lr_S001.csv`, with a separate provenance
  record under `outputs/S001-resnet18-submission-v1/`.
- This is inference from an existing run, not a new training experiment or a new
  validation measurement; do not add an invented training row to results.csv.

## Command

```powershell
.venv/Scripts/python.exe -m src.predict --checkpoint checkpoints/resnet18_low_lr_best.pt --data-dir data --output submissions/final/resnet18_low_lr_S001.csv --device cpu
```

## Outcome

Completed local CPU inference for all 1,340 test images. The CSV has exactly
`id,label` columns, unique complete sample IDs in the supplied order, no missing
values, and only the 67 permitted class labels. Inference probabilities were
checked for finiteness and expected shape before label conversion.

- Submission: `submissions/final/resnet18_low_lr_S001.csv`.
- CSV SHA-256: `c832ea4b801e0bb1a6254798ea744910f36a6704547a65f68dc3a89f63d42c49`.
- Checkpoint SHA-256: `811ccd58c8d283764310f33035a7e43461b5a601c8271a3a55a3d3930ee2b9e2`.
- Code base: `719d3f8a1958759fd9f7fe20c3f25b4b42365491`, with uncommitted
  checkpoint-compatibility, prediction validation, tests, and documentation changes.
- Local provenance: `outputs/S001-resnet18-submission-v1/submission_metadata.json`.
- Tests: 14 of 15 project checks passed, including all new ResNet18 tests and
  existing ResNet50 submission tests. The unchanged notebook 03 contains stored
  widget/output metadata and fails the existing notebook-source check; that
  unrelated notebook content was not changed in this task.
- No new fitting, validation-score claim, or Kaggle upload. The 68.1989% figure
  is the source run's validation accuracy, not a Kaggle leaderboard score.

Upload the CSV file on the competition's submission page. Do not upload the
`.pt` checkpoint in place of the prediction CSV. If the team supplies E001's
77.7674% checkpoint, generate a separate CSV with the shared prediction command;
this artifact is from the best checkpoint currently available on this machine.
