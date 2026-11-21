# Shared train/validation split

Use `train_ids.csv` for fitting and `val_ids.csv` for model selection. Join these
IDs to `data/raw/train.csv` with a one-to-one merge to obtain paths and labels.
Never include validation IDs in model fitting for validation experiments.

Regenerate and validate from the repository root:

```bash
python scripts/prepare_data.py
```

The script reads seed 42 and a 20% validation fraction from `config.yaml`, sorts
training records by ID, and uses a stratified scikit-learn split. It checks the
provided manifests and decodes all images before saving split assignments.
It does not obtain test labels or use test images to fit a model.

- `train_ids.csv`: IDs used for fitting.
- `val_ids.csv`: IDs reserved for validation.
- `class_counts.csv`: Per-class total, fitting, and validation counts.
- `metadata.json`: Seeds, counts, scikit-learn version, and manifest/split hashes.

Commit these files so teammates use the same assignments. Use the saved IDs
rather than independently generating new splits. The script refuses to replace
existing files with different content; use a separate `--output-dir` to review
a deliberately changed split. Dataset images remain ignored by Git.

The hashes identify CSV manifests and assignments, not the image bytes.
`config_sha256` records the full config at the time this split was created. If
only unused config text changes and all computed split metadata and assignments
remain identical, validation preserves the original creation-time metadata.
Changes to seeds, fractions, data, assignments, or library versions still require
review in a separate output directory. The original full config is available in
Git at the split-creation commit; the current root config contains only settings
used by data preparation.
