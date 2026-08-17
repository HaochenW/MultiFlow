# Release guide

Before publishing, confirm institutional intellectual-property requirements
and add the paper citation when it becomes available.

## 1. Create the GitHub repository

From this clean directory only:

```bash
git init
git add .
git commit -m "Initial public MultiFlow release"
git branch -M main
git remote add origin https://github.com/HaochenW/MultiFlow.git
git push -u origin main
```

Do not copy Git history from the 22 GB experimental directory without a
separate history and secret scan.

## 2. Validate locally

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m build
twine check dist/*
```

Install the wheel into a new environment and run the CLI smoke test before
tagging a release.

## 3. Configure Trusted Publishing

On TestPyPI and PyPI, create a pending Trusted Publisher for the GitHub owner,
repository `MultiFlow`, workflow filenames `testpypi.yml` and `publish.yml`,
and environments `testpypi` and `pypi`, respectively. No long-lived API token
is required.

Run the TestPyPI workflow manually first. After verifying installation from
TestPyPI, create a GitHub release tagged `v0.1.0`; the production workflow is
triggered only by a published GitHub release.

## 4. Store large assets elsewhere

H5AD/H5MU files, latent NPZ files, encoder checkpoints, generated results, and
trained weights are excluded from Git and the wheel. Publish permitted model
weights through Zenodo or a GitHub Release, provide SHA256 checksums, and link
them from a model card.
