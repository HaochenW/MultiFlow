# MultiFlow

[![CI](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Start here:** [Full tutorial](docs/tutorial.md) ·
[H5MU contract](docs/h5mu_contract.md) ·
[Model contract](docs/model_contract.md) ·
[Implementation audit](docs/implementation_audit.md)

MultiFlow learns a coupled vector field for paired single-cell RNA and ATAC
states. It supports cell-type-conditioned generation and
perturbation-conditioned prediction while keeping the two modalities paired.

> **Alpha release.** The flow model and raw-H5MU paper workflow are ready for
> testing. The processed perturbation dataset is not yet hosted publicly; the
> tutorial therefore requires a local path for that dataset.

## Install

Install the current GitHub version:

```bash
python -m pip install "git+https://github.com/liuq-lab/MultiFlow.git"
```

After the first PyPI release, the shorter installation command will be:

```bash
python -m pip install multiflow-omics
```

The installed command is simply `multiflow`. The PyPI distribution uses the
longer name because `multiflow` is already registered by an unrelated project.

## Start with the tutorial

The user-facing workflow starts from raw paired profiles. It trains the
task-specific encoder(s), trains MultiFlow, samples paired latent states and
decodes them back to RNA/ATAC profiles:

```text
raw H5MU -> VAE/AE -> MultiFlow -> matching decoders -> profile H5MU
```

The [full tutorial](docs/tutorial.md) gives complete commands for:

- cell-type-conditioned generation on the public OpenProblem data; and
- leave-one-cell-type-out perturbation prediction on processed GSE274113.

The small latent-only example remains available for package smoke testing:

```bash
multiflow data example --output toy_multiflow.h5mu
multiflow data validate toy_multiflow.h5mu
```

## H5MU contract

The initial user input is one paired raw-profile file:

```text
paired.h5mu
├── rna.X                         raw RNA counts
├── atac.X                        binary ATAC accessibility
└── rna.obs["cell_type"]          biological condition
```

`multiflow paper encode` creates a derived H5MU containing the two
`X_multiflow` representations. RNA and ATAC `obs_names` must be identical and
in the same order. See [the complete H5MU contract](docs/h5mu_contract.md).

## OpenProblem data

The paired scDiffusion-X OpenProblem dataset can be downloaded directly:

```bash
multiflow data download openproblem \
  --output data/openproblem_filtered.h5mu \
  --accept-license
```

This is a version-pinned **8.38 GB** download from Figshare. The downloader
supports resuming, checks the expected file size and MD5, and publishes the
file only after verification.

- DOI: [10.6084/m9.figshare.28582061.v3](https://doi.org/10.6084/m9.figshare.28582061.v3)
- Source: [scDiffusion-X](https://github.com/EperLuo/scDiffusion-X)
- License: CC BY 4.0

The upstream file contains the correct raw RNA counts and binary ATAC
profiles. The tutorial splits it, trains the RNA VAE and ATAC AE, and runs
`multiflow paper encode`; users do not create latent arrays manually.

## Models

- `cell-state` (default): bidirectional cross-attention MultiFlow.
- `perturbation`: adds context and perturbation embeddings.
- `concat`: concatenation architecture retained for benchmark reproduction.

The latent flow objective, normalization, and sampling contracts are described
in [docs/model_contract.md](docs/model_contract.md).

The source-to-release implementation checks are summarized in
[docs/implementation_audit.md](docs/implementation_audit.md).

## Python API

```python
from multiflow_omics import MultiFlow, TrainingConfig, fit

model = MultiFlow(rna_dim=128, atac_dim=128, num_classes=4)
result = fit(
    model,
    rna_latents,
    atac_latents,
    labels=cell_type_codes,
    config=TrainingConfig(epochs=600, seed=0),
)
```

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[docs/releasing.md](docs/releasing.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The paper DOI
will be added when available.
