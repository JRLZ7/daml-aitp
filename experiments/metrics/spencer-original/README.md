# Spencer original experiment results

The three `baseline_*.csv` files are unchanged copies from Spencer Kolodny's source commit
`859926a83d2b17bed0641fb9b5560038671a0a18` on `skolodny/initialTesting`.
Original notebooks and their authored history remain in Git. The source split
fingerprint is retained as originally recorded; shared validation membership was
independently verified to match all 1,066 IDs and ground-truth labels.

Historical scores: most-frequent 1.5009%; RGB thumbnail/histogram + linear SVM
8.3490%. These are validation results. A shared-workflow rerun is recorded under
E010 with its own code commit, environment, and metrics; do not edit these CSVs
to match reruns. There are no image files or embedded image outputs here.

## Original ResNet18 experiments

The ten `resnet18_*.csv` files were relocated from the top-level `experiments/`
directory on 2026-11-21. Every moved file was verified byte-for-byte using SHA-256.
The move preserves the recorded results and does not represent a new run.

- `resnet18_*` (without `low_lr`): original ImageNet ResNet18 transfer-learning
  run, 65.3846% validation accuracy and 0.6506 macro F1.
- `resnet18_low_lr_*`: five-epoch fine-tuning experiment at `1e-5`, selecting
  epoch 5 with 68.1989% validation accuracy and 0.6801 macro F1.

These historical experiments use the split fingerprints and label mappings
recorded in their CSVs. Notebooks 03 and 04 now read and write their CSVs here,
including comparisons against the original classical baseline. This relocation
does not adapt those historical notebooks to the merged shared loader or rerun
their training. The shared `experiments/results.csv` index remains in place.
Codex performed the file relocation, path updates, output clearing, and checks.
