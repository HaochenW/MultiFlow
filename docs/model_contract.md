# Model and data contract

## Canonical architectures

`MultiFlow` is an alias of `CellStateFlow`, the paired cell-state generator
used by the cross-attention experiments. RNA and ATAC are processed in two
streams. Three bidirectional feature-level cross-attention modules allow each
modality's velocity to depend on the other modality's current state.

`MultiFlowPerturbation` adds cell-context embeddings and perturbation
embeddings to the same coupled vector field. Perturbation index `0` is the
control condition and contributes an exactly zero perturbation embedding.

`MultiFlowConcat` is the concatenation ablation used by later generation
experiments. It is retained for reproducibility but is not the canonical
cross-attention architecture.

## Latent inputs

All training inputs must satisfy the following contract:

- RNA and ATAC are finite `float32`-compatible matrices;
- rows are paired cells in identical order;
- label arrays use zero-based, contiguous integer indices;
- the context matrix row order equals the cell-context label order;
- encoder feature order and preprocessing scale are fixed outside this
  package; and
- normalization statistics are estimated on training cells only.

The package standardizes each latent feature independently by default. The
standardizer is stored in the portable checkpoint and inverted after sampling.
It does not replace the encoder-specific scale factors required by an external
decoder.

## Flow objective

For target latent state `x1`, Gaussian source `x0`, and time `t ~ U(0, 1)`,
MultiFlow trains on the straight-line path

```text
x_t = (1 - t) x0 + t x1
v_t = x1 - x0
```

and minimizes the sum of RNA and ATAC mean-squared velocity errors.
Cell-state and perturbation models use independent Gaussian sources for the
two modalities. The concat ablation uses one shared source and therefore
requires equal latent dimensions.

## Sampling protocol

The default `rng_mode="legacy_interleaved"` matches the research sampler's
RNA/ATAC random-draw order. Record `seed`, `batch_size`, ODE `steps`, device,
and package version because all are part of exact stochastic reproduction.
The perturbation experiments used 50 midpoint steps; the general CLI default
is 100.

`rng_mode="batch_invariant"` makes a fixed seed independent of batch size by
allocating the complete source state first. This option can use substantially
more device memory for very large sample counts.

## Out of scope for version 0.1

Version 0.1 does not bundle the dataset-specific scDiffusion/scDiffusion-X
encoders and decoders. They require tightly pinned external releases and
distinct raw-count/log-normalized input contracts. Keeping them outside the
core wheel prevents hidden preprocessing, feature-order changes, and fragile
default dependencies. A future optional adapter package can add audited H5MU
end-to-end workflows.
