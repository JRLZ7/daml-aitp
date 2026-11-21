# Competition data

Download only the course competition files. Do not use the public Indoor67 mirror
or original labels. Each teammate downloads their own copy; competition images
must stay out of Git and the final source package.

From the repository root, with the virtual environment activated:

1. Join the private course competition and accept its rules in Kaggle.
2. Generate an API token at https://www.kaggle.com/settings/api.
3. Run the following for the course competition:

```bash
python scripts/download_data.py daml-aitp-fall-26-challenge --login
```

The login prompt accepts your token privately. Do not paste tokens into source,
notebooks, chat, or terminal commands. Subsequent downloads can omit `--login`
when credentials are already configured. Authentication and competition access
are required; a 403 can mean rules have not been accepted or access is missing.

Official API documentation: https://github.com/Kaggle/kagglehub

Files download into `data/raw/` (ignored by Git). Inspect the result before
extracting any remaining image archive. Arrange the files as follows:

```text
data/raw/
  train.csv
  test.csv
  classes.csv
  sample_submission.csv
  images/
    train/
    test/
```

Image paths resolve as `data/raw/images/` plus the CSV's `file_path` value.
Do not add another `train/` or `test/` prefix to that value.

Expected course data: 5,327 labeled training rows, 1,340 test rows, 67 classes.
Validate these counts and all referenced image paths before creating a split.
Save split IDs outside `data/`, under the tracked `splits/` directory.

For a different storage location, pass `--output-dir /path/to/data`.
