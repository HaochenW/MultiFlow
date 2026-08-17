# H5MU data contract

MultiFlow uses H5MU as the public container for paired cells. The flow itself
is trained on modality-specific latent states; raw matrices remain in the same
file for provenance and later decoding.

## Required structure

An input must contain `rna` and `atac` modalities. Their `obs_names` must be
unique and match exactly in both value and order. By default:

- `rna.X` contains nonnegative integer raw counts;
- `atac.X` contains only 0 and 1;
- `rna.obs["cell_type"]` contains a non-missing cell-type name;
- `rna.obsm["X_multiflow"]` contains finite RNA latents; and
- `atac.obsm["X_multiflow"]` contains finite ATAC latents.

Use `--rna-representation` and `--atac-representation` to select another
explicit `obsm` key. `X` can be selected explicitly, but is never used as an
implicit fallback when a latent key is absent.

## Scale boundaries

The core package does not guess preprocessing from numeric ranges. The paper
RNA VAE and multimodal AE use different input and decoder scales, so an encoder
release must state all of the following:

1. input matrix/layer and scale;
2. feature identifiers in exact order;
3. checkpoint and configuration hashes;
4. latent dimension and encoder-specific scale factors;
5. decoder output scale; and
6. whether the encoder was trained only on the training split.

MultiFlow then fits an additional feature-wise latent standardizer using the
training cells. That standardizer and the string-to-integer condition mapping
are stored in `model.pt` and `run.json`.

For perturbation training, `obs["perturbation"]` (or the selected column) must
contain the named control condition. The CLI reserves that condition as index
0 and therefore as the exact zero perturbation vector. The context embedding
matrix is stored in `uns["multiflow_context_matrix"]`; its row names must be
stored in `uns["multiflow_context_classes"]` and must exactly match the fixed
cell-type class order. MultiFlow rejects missing or reordered row labels.

## Generated files

Version 0.1 writes paired generated **latent** states to `rna.X` and `atac.X`
with `uns["matrix_scale"] = "latent"`. `mdata.uns["multiflow"]` records the
checkpoint SHA256, model configuration, seed, ODE steps, batch size, label
mapping, RNG mode, and whether latent standardization was inverted.

Decoded gene/peak profiles require the exact encoder bundle used to create the
training latents. They must not be inferred from a generic H5MU file.
