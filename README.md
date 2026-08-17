# MultiFlow

[![CI](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/liuq-lab/MultiFlow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MultiFlow learns a coupled vector field for paired single-cell RNA and ATAC
states. It supports cell-type-conditioned generation and
perturbation-conditioned prediction while keeping the two modalities paired.

> **Alpha release.** The flow model and portable H5MU interface are ready for
> testing. Paper encoder checkpoints will be released separately after their
> feature order and scale contracts are finalized.

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

## Quick start

Create a small paired H5MU example:

```bash
multiflow data example --output toy_multiflow.h5mu
```

Validate cell pairing, raw scales, and latent representations:

```bash
multiflow data validate toy_multiflow.h5mu
```

Train a cell-type-conditioned model:

```bash
multiflow train \
  --input toy_multiflow.h5mu \
  --output runs/toy \
  --epochs 20 \
  --device cpu
```

Generate paired states using a cell-type name:

```bash
multiflow generate \
  --run runs/toy \
  --output generated_B_cell.h5mu \
  --cell-type "B cell" \
  --n 100 \
  --device cpu
```

The run directory contains `model.pt`, `history.csv`, and a readable
`run.json`. The generated H5MU stores paired RNA/ATAC latent states, the fixed
cell-type mapping, sampling seed, ODE steps, and checkpoint checksum.

## H5MU contract

MultiFlow uses one file instead of separate anonymous arrays:

```text
paired.h5mu
├── rna.X                         raw RNA counts
├── atac.X                        binary ATAC accessibility
├── rna.obs["cell_type"]          biological condition
├── rna.obsm["X_multiflow"]       RNA latent state
└── atac.obsm["X_multiflow"]      ATAC latent state
```

RNA and ATAC `obs_names` must be identical and in the same order. Raw `X` is
never silently substituted for a missing encoder representation. See
[the complete H5MU contract](docs/h5mu_contract.md).

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

The upstream file contains raw RNA counts and binary ATAC profiles, but not
MultiFlow encoder latents. Run `multiflow data validate` to inspect it. To
reproduce the paper pipeline, add latents produced by the exact released RNA
and ATAC encoder bundle; do not train directly on the raw feature matrices.

## Models

- `cell-state` (default): bidirectional cross-attention MultiFlow.
- `perturbation`: adds context and perturbation embeddings.
- `concat`: concatenation architecture retained for benchmark reproduction.

The latent flow objective, normalization, and sampling contracts are described
in [docs/model_contract.md](docs/model_contract.md).

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
