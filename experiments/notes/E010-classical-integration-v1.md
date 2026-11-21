# E010 — Reproduce Spencer's classical baseline in the shared workflow

Date planned: 2026-11-21. Initial status: predeclared integration verification. Completed; outcome below.

Spencer authored exploration and RGB-thumbnail/histogram + linear-SVM work in
`859926a83d2b17bed0641fb9b5560038671a0a18`. Jason's branch supplies the verified
shared loader, split, training/evaluation infrastructure, and neural experiments.
Codex adapts the classical method to the common interfaces and executes this
integration rerun. This attribution does not imply other team contributions.

Decision D012. Fixed settings: `configs/classical.yaml`, 16x16 RGB thumbnail plus
16-bin histograms per RGB channel (816 dimensions), training-only StandardScaler,
LinearSVC(C=0.01, dual=False, max_iter=5000, seed=42). Also reproduce the
most-frequent-class dummy. No tuning, test fitting, or pretrained weights.

Use `splits/train_ids.csv` and `splits/val_ids.csv` and the explicit classes.csv
order. Spencer's committed prediction IDs and true labels were checked: the
validation membership matches all 1,066 shared IDs exactly. Do not regenerate it.

```bash
python -m src.classical --config configs/classical.yaml --run-id E010-classical-integration-v1 --owner "Codex (AI-assisted integration execution)"
```

Notebook 02 calls this same runner when the local run does not exist, then plots
the recorded results. Outputs are generated locally and never stored in the
tracked notebook. The two result rows use `-dummy` and `-svm` suffixes.

Expected historical references, not promised rerun scores: dummy 1.5009%, SVM
8.3490%. Preserve the original CSVs under `experiments/metrics/spencer-original/`.
Rerun numbers must be recorded separately with actual code/environment provenance.

Verification plan: run all pipeline tests, execute both notebooks headlessly,
verify checkpoint reload predictions, and compare results with the original CSVs.
The combined ResNet-50 and regularization results must remain unchanged.

## Outcome

Executed from clean integration merge commit
`cd2ce318f8d5e271d4007c6eda779ba9bb21b186`, with both Jason and Spencer histories
preserved. No settings were tuned during verification.

| Model | Validation accuracy | Correct / 1,066 | Macro F1 |
| --- | --- | --- | --- |
| Most frequent | 1.5009% | 16 | 0.0004414 |
| RGB thumbnail/histogram + linear SVM | 8.3490% | 89 | 0.0713338 |

All 1,066 SVM predicted labels and true labels exactly match Spencer's original
CSV after aligning by ID. The original CSV archive is byte-identical to his
source commit. Feature extraction took 21.17 seconds and fitting 59.09 seconds;
total suite time was 1.338 minutes. No warnings were recorded. Both locally saved
joblib pipelines reproduced their validation predictions after reloading.

Twelve unit/source checks pass. Both notebooks executed headlessly: exploration
ran 8 code cells and produced 3 charts; classical analysis ran 7 code cells and
produced 2 charts. Executed copies with images remain local in ignored outputs;
tracked notebooks contain source only. Neural code/configs and all shared split
files are unchanged from Jason's source branch. The 77.77% ResNet baseline remains
the strongest recorded model.

Compact rerun results are in `experiments/metrics/E010-classical-integration-v1/`.
Full artifacts and environment are local under `outputs/E010-classical-integration-v1/`.
The new rows use `-dummy` and `-svm` suffixes; the recorded suite runtime covers
shared feature extraction/fitting and is not meant to be added across both rows.

This confirms interface and method compatibility, not an independent evaluation
of generalization: it uses the same fixed validation set and the original recipe.
No test predictions or Kaggle submissions were made.
