# MultiFlow

MultiFlow learns a conditional vector field over **paired RNA and ATAC latent
states**. The package contains the latent-level model, flow-matching training,
ODE sampling, and portable checkpoint utilities used by the MultiFlow method.

> **Pre-release notice:** this source tree is an alpha research package. Review
> the release checklist below before publishing a new version.

## Scope

MultiFlow expects paired, finite, two-dimensional latent arrays whose rows
refer to the same cells in the same order. It does **not** normalize raw counts,
select features, or bundle RNA/ATAC encoders and decoders. Those steps remain
explicit so that feature order, input scale, model provenance, and training
scope can be audited independently.

The command-line interface accepts an NPZ file with:

- `rna`: RNA latents with shape `(n_cells, rna_dim)`;
- `atac`: ATAC latents with shape `(n_cells, atac_dim)`;
- `labels` (optional): zero-based cell-context labels;
- `perturbations` (perturbation model): zero-based perturbation labels, with
  label `0` reserved for control; and
- `context_matrix` (perturbation model): one context embedding per label.

The default training command standardizes each modality feature-wise using
statistics estimated from the supplied training latents. Sampling reverses
that standardization before writing the output unless
`--keep-standardized` is passed.

## Which model is MultiFlow?

The public `MultiFlow` API is the paired, bidirectional cross-attention model
used in the cell-state experiments. `MultiFlowPerturbation` extends the same
coupled vector field with cell-context and perturbation embeddings.
`MultiFlowConcat` is retained as an explicitly named ablation; it is not
presented as the cross-attention architecture. See
[docs/model_contract.md](docs/model_contract.md) for the exact contracts.

## Installation

From a source checkout:

```bash
python -m pip install -e .
```

For tests and release checks:

```bash
python -m pip install -e ".[dev]"
```

Create a small example input if desired:

```bash
python examples/make_toy_latents.py
```

## Train and sample latent states

Train a cell-state model:

```bash
multiflow-omics train-latents \
  --input paired_train_latents.npz \
  --output checkpoints/multiflow.pt \
  --model cell-state \
  --epochs 600
```

Generate paired latent states for cell-context label `2`:

```bash
multiflow-omics sample-latents \
  --checkpoint checkpoints/multiflow.pt \
  --output generated_label_2.npz \
  --n 1000 \
  --label 2 \
  --steps 100
```

Inspect checkpoint metadata:

```bash
multiflow-omics inspect-checkpoint --checkpoint checkpoints/multiflow.pt
```

Convert a historical research checkpoint before publishing it:

```bash
multiflow-omics migrate-checkpoint \
  --source trusted_legacy_checkpoint.pt \
  --output checkpoints/multiflow_v1.pt
```

Migration uses PyTorch's safe weights-only loader by default. Only add
`--trust-source` for a legacy pickle whose provenance you have independently
verified. Raw historical state dictionaries often lack latent normalization
statistics; supply the four original training-only arrays with
`--standardizer-npz` before using decoded outputs.

The output NPZ contains paired `rna` and `atac` latent arrays. Decode them with
the exact encoder/decoder release and feature order used to create the training
latents.

## Python API

```python
from multiflow_omics import MultiFlow, TrainingConfig, fit, seed_everything

seed_everything(0)  # seed before model construction
model = MultiFlow(
    rna_dim=rna_latents.shape[1],
    atac_dim=atac_latents.shape[1],
    num_classes=n_cell_types,
)
result = fit(
    model,
    rna_latents,
    atac_latents,
    labels=cell_type_codes,
    config=TrainingConfig(epochs=600, seed=0),
)
```

The default sampler uses the historical batched random-draw order. For exact
reproduction, record both `seed` and `batch_size`. An optional
`rng_mode="batch_invariant"` is available when identical samples across batch
sizes are more important than minimizing device memory.

## Reproducibility requirements

For a published analysis, record all of the following outside the checkpoint:

1. dataset accession and immutable split manifest;
2. RNA and ATAC feature names in order;
3. preprocessing and encoder input scales;
4. encoder/decoder versions and hashes;
5. latent dimensions and training-only normalization statistics;
6. label-to-integer mappings; and
7. package version, seed, optimizer settings, and sampling steps.

Do not publish cells, raw data, private paths, access tokens, trained weights,
or dataset-derived embeddings unless their data-use terms permit it.

## Development

```bash
ruff check .
pytest
python -m build
twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The complete release
procedure is in [docs/releasing.md](docs/releasing.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Replace the
temporary developer-group entry with the final author list and add the paper
DOI before the first public release.


