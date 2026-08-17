# Implementation audit

This document records the source-to-release checks performed before writing the
public tutorial. It separates the executed model graph from inactive notebook
experiments and from preprocessing that remains outside the package.

## Reference hierarchy

1. Cell-state architecture: executed cells 10, 14, 15, and 19 in
   `MultiFlow/generation/best_record/Joint_model_VAE_AE_encoder_cross_attention.ipynb`.
2. Perturbation architecture: the production leave-one-cell-type-out runner
   `run_leave_one_out_all_celltypes.py`, which incorporates the intended zero
   control embedding missing from the earlier exploratory notebook cell.
3. Concatenation ablation: the active graph in
   `Joint_model_VAE_diffusion_encoder_cross_attention_concat.ipynb`, represented
   by the corrected research module `multiflow_generation/model.py`.
4. Protocol defaults and encoder boundary: manuscript Methods plus the executed
   training and sampling cells.

## Numerical graph checks

The release models were initialized with the reference state tensors after a
semantic key mapping and evaluated on identical random inputs. The largest
absolute output difference was `0.0` for all three checks:

| Release model | Reference | State entries | Max. absolute difference |
|---|---|---:|---:|
| `CellStateFlow` | executed cross-attention notebook module | 87 | 0.0 |
| `PerturbationFlow` | production leave-one-out runner | 88 | 0.0 |
| `ConditionalConcatFlow` | corrected concat notebook module | 47 | 0.0 |

The executable audit is kept outside the package test suite because it points
to the local research tree. The package tests independently cover model
construction, training, sampling, checkpoint migration, H5MU validation, and
CLI round trips.

## Training and sampling checks

- target path: linear interpolation between Gaussian source and encoded target;
- objective: RNA MSE plus ATAC MSE with equal weight;
- source noise: independent for the cross-attention and perturbation models;
- concat ablation source: one shared Gaussian state for both modalities;
- optimization: Adam, learning rate `1e-4`, batch size 512, 600 epochs;
- latent normalization: per-feature mean and standard deviation estimated from
  training cells and inverted after sampling;
- generation solver: midpoint ODE integration with 100 steps;
- perturbation solver: midpoint ODE integration with 50 steps; and
- perturbation control: fixed class index 0 with an exact zero embedding.

## Explicit boundaries and corrected exploratory cells

- The public package begins with paired encoder latents. It does not yet bundle
  the paper RNA VAE, multimodal ATAC autoencoder, decoders, feature files, or
  encoder scale factors.
- The package does not create leave-one-cell-type-out splits or apply
  benchmark-specific latent/profile corrections. These operations must remain
  explicit and training-only.
- The exploratory concat notebook assigns `x1_atac = rna_latent` in one cell.
  This is inconsistent with paired RNA/ATAC training and with the surrounding
  code. The corrected research module and release use `atac_latent`; the concat
  model is labeled as an ablation rather than the canonical paper model.
- The earliest perturbation notebook initialized the control embedding as a
  trainable random vector. The later production runner fixes and masks row 0,
  matching the manuscript definition `p = 0`; the release follows that
  production behavior.
- The previous CLI fallback of 200 ODE steps did not match the paper protocol.
  The release now selects 100 steps for cell-state generation and 50 for
  perturbation unless an explicit override is recorded.

These boundaries are also stated in the README and tutorial so users cannot
mistake generated latent states for decoded biological profiles.
