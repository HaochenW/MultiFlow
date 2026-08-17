# Contributing

MultiFlow is an alpha research package. Please discuss substantial API or
model changes in an issue before opening a pull request.

## Development checks

Create an isolated environment and install the development dependencies:

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
twine check dist/*
```

Changes should include focused tests and documentation. Public APIs must use
portable inputs and must not depend on lab-specific paths, schedulers, private
datasets, or external source checkouts.

## Data and provenance

Do not commit human-subject data, generated cells derived from restricted
datasets, trained checkpoints, access tokens, personal identifiers, or server
paths. Use synthetic arrays in tests. Describe the input scale, feature order,
data split, encoder/decoder version, and source license for changes that affect
scientific results.

Contributors must disclose code adapted from another project and preserve all
required copyright and license notices. Contributions are accepted under the
MIT License used by this repository.
