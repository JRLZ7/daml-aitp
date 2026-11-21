# E010 — Reproduce Spencer's classical baseline in the shared workflow

Date planned: 2026-11-21. Status: predeclared integration verification.

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

Pending execution. Record deviations rather than changing settings to match a score.
