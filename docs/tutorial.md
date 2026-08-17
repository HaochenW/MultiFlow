# MultiFlow tutorial

This tutorial covers the public MultiFlow latent-flow package. It starts with a
small H5MU file, then shows the exact input contract for a real paired dataset.

> **Release scope.** MultiFlow trains and samples paired RNA/ATAC **latent
> states**. Version 0.1 does not bundle the dataset-specific RNA VAE, ATAC
> autoencoder, or profile decoders used in the paper. The package therefore
> never substitutes raw counts for missing latents and never labels latent
> output as gene expression or peak accessibility.

## 1. Install

Create a clean Python environment and install the current GitHub version:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/liuq-lab/MultiFlow.git"
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux and macOS. Confirm the installation:

```bash
multiflow --version
```

Use a CUDA-enabled PyTorch installation for real datasets. The toy example
below runs on CPU.

## 2. Run the five-minute example

Create a small paired H5MU file. It contains raw RNA counts, binary ATAC values,
paired cell identifiers, cell types, and four-dimensional toy latents.

```bash
multiflow data example --output toy_multiflow.h5mu
multiflow data validate toy_multiflow.h5mu
```

The validation report should contain:

```text
"paired_raw_contract_ok": true
"latent_ready": true
```

Train a small cell-type-conditioned model:

```bash
multiflow train \
  --input toy_multiflow.h5mu \
  --output runs/toy \
  --model cell-state \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 32 \
  --device cpu
```

Generate 100 paired latent states for one cell type:

```bash
multiflow generate \
  --run runs/toy \
  --output generated_B_cell.h5mu \
  --cell-type "B cell" \
  --n 100 \
  --device cpu
```

The run directory contains `model.pt`, `history.csv`, and `run.json`. The
generated H5MU records the condition, seed, solver steps, checkpoint checksum,
and latent standardization provenance.

## 3. Prepare a real H5MU file

MultiFlow requires one paired container:

```text
paired.h5mu
├── rna.X                         raw RNA counts
├── atac.X                        binary accessibility
├── rna.obs["cell_type"]          cell-type condition
├── rna.obsm["X_multiflow"]       RNA encoder latent
└── atac.obsm["X_multiflow"]      ATAC encoder latent
```

The RNA and ATAC `obs_names` must be identical and in the same order. The
paper models use 128-dimensional latents for both modalities. RNA encoder
preprocessing is total-count normalization to 10,000 followed by `log1p`; the
ATAC encoder receives the binary accessibility matrix. Encoder feature order,
weights, and preprocessing must remain fixed between training and decoding.

If the encoder outputs are already in memory, store them directly in H5MU:

```python
import mudata as mu
import numpy as np

mdata = mu.read_h5mu("paired_raw.h5mu")
rna_latent = np.asarray(rna_latent, dtype=np.float32)
atac_latent = np.asarray(atac_latent, dtype=np.float32)

assert rna_latent.shape == (mdata.n_obs, 128)
assert atac_latent.shape == (mdata.n_obs, 128)

mdata["rna"].obsm["X_multiflow"] = rna_latent
mdata["atac"].obsm["X_multiflow"] = atac_latent
mdata.write_h5mu("paired_multiflow.h5mu")
```

This snippet only stores encoder outputs; it does not define or replace the
paper encoders. Validate the completed file before training:

```bash
multiflow data validate paired_multiflow.h5mu --json validation.json
```

### OpenProblem download

Download the paired scDiffusion-X OpenProblem data when a real raw-data example is needed:

```bash
multiflow data download openproblem \
  --output data/openproblem_filtered.h5mu \
  --accept-license
```

The file is 8.38 GB and is pinned to Figshare record
`10.6084/m9.figshare.28582061.v3`. It contains raw RNA counts and binary ATAC
profiles, but not MultiFlow encoder latents. A validation result with
`latent_ready: false` is therefore expected until the audited encoder outputs
are added.

## 4. Train the paper cell-state flow

The following values match the cell-state training protocol: two modality
branches, three bidirectional cross-attention modules, hidden width 512, two
residual blocks per path, 600 epochs, batch size 512, and Adam with learning
rate `1e-4`.

| Setting | Cell-state generation | Perturbation prediction |
|---|---:|---:|
| RNA / ATAC latent width | 128 / 128 | 128 / 128 |
| Hidden width | 512 | 512 |
| Residual blocks per path | 2 | 2 |
| Cross-attention feature width | 64 | 64 |
| Training epochs | 600 | 600 |
| Training batch size | 512 | 512 |
| Adam learning rate | 1e-4 | 1e-4 |
| Gaussian source | independent by modality | independent by modality |
| Midpoint ODE steps | 100 | 50 |
| Sampling batch size | 512 | 256 |

```bash
multiflow train \
  --input paired_multiflow.h5mu \
  --output runs/cell_state \
  --model cell-state \
  --condition-key cell_type \
  --epochs 600 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --hidden-dim 512 \
  --num-blocks 2 \
  --cross-attention-dim 64 \
  --sampling-steps 100 \
  --device cuda \
  --seed 0 \
  --hash-input
```

By default, each RNA and ATAC latent feature is standardized with statistics
estimated from the supplied training cells. The statistics are saved in the
checkpoint and inverted after sampling. Do not use `--no-standardize` for the
paper protocol.

## 5. Generate a cell type

Use a biological name from the training H5MU rather than an anonymous integer:

```bash
multiflow generate \
  --run runs/cell_state \
  --output generated_cd4_t.h5mu \
  --cell-type "CD4+ T activated" \
  --n 1000 \
  --steps 100 \
  --batch-size 512 \
  --device cuda \
  --seed 0
```

Sampling starts from independent RNA and ATAC Gaussian states and uses the
midpoint ODE solver. The output is in the original encoder latent scale because
the saved feature standardization is inverted automatically. Apply the same
RNA and ATAC decoders used to create the training latents to obtain biological
profiles.

## 6. Perturbation-conditioned flow

Perturbation training needs a **training-only** H5MU file. A leave-one-cell-type
out experiment must remove the held-out perturbed cells before computing any
standardization or context statistics. In addition to paired latents, the file
must contain:

```text
rna.obs["cell_type"]
rna.obs["perturbation"]
mdata.uns["multiflow_context_matrix"]
mdata.uns["multiflow_context_classes"]
```

The context matrix contains one concatenated RNA/ATAC mean latent vector per
cell type, computed from the training split. Its row order must exactly equal
the string array in `multiflow_context_classes`. The perturbation named
`control` is always assigned index 0 and contributes a zero perturbation
embedding.

Train the perturbation flow:

```bash
multiflow train \
  --input perturbation_train_multiflow.h5mu \
  --output runs/perturbation \
  --model perturbation \
  --condition-key cell_type \
  --perturbation-key perturbation \
  --control-perturbation control \
  --epochs 600 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --hidden-dim 512 \
  --num-blocks 2 \
  --cross-attention-dim 64 \
  --sampling-steps 50 \
  --device cuda \
  --seed 0
```

Generate one held-out context and perturbation:

```bash
multiflow generate \
  --run runs/perturbation \
  --output generated_heldout.h5mu \
  --cell-type "CD4+ T activated" \
  --perturbation "STAT1" \
  --n 500 \
  --steps 50 \
  --batch-size 256 \
  --device cuda \
  --seed 0
```

The public package fits and samples the paired perturbation flow, but does not
automatically create leave-one-cell-type-out splits, run the external
encoders/decoders, or apply benchmark-specific post-processing. Keep those
steps explicit so held-out data cannot leak into training statistics.

## 7. Reproducibility checklist

Before comparing runs, confirm that all items below are unchanged:

- paired cell order and feature order;
- RNA normalization and ATAC binarization used by the encoders;
- encoder and decoder checkpoint hashes;
- latent dimensions and H5MU representation keys;
- label and perturbation class mappings in `run.json`;
- training split used to estimate latent means and standard deviations;
- model type, epoch count, batch size, learning rate, and random seed; and
- solver steps, generation batch size, and sampling seed.

For a strict stochastic replay, keep the same `--batch-size` and use the
default `legacy_interleaved` RNG mode. Use `--rng-mode batch_invariant` only
when reproducibility across different sampling batch sizes is more important
than matching the original random-draw order.

## 8. Common errors

**`latent_ready` is false.** One or both `X_multiflow` representations are
missing, non-finite, or have the wrong number of rows.

**Cell identifiers do not match.** Reorder both modalities to the same paired
`obs_names`; do not rely on implicit row alignment.

**Unknown cell type or perturbation.** Inspect `run.json` and use the exact
stored biological name.

**CUDA memory error.** Reduce generation `--batch-size`. For an exact replay,
remember that the default RNG mode treats batch size as part of the stochastic
protocol.

**Need gene/peak profiles.** The generated file contains latent states. Decode
with the same audited encoder bundle used for training; version 0.1 does not
ship a generic decoder.
