# MultiFlow tutorial

MultiFlow has two tasks:

1. **cell-state generation**: generate paired RNA and ATAC profiles for a cell
   type; and
2. **perturbation prediction**: predict paired RNA and ATAC responses for a
   held-out cell type and perturbation.

Both workflows start from biological profiles, not precomputed latent arrays:

```text
raw paired H5MU
    -> task-specific RNA and ATAC encoders
    -> paired 128-dimensional latent states
    -> MultiFlow
    -> the same task-specific decoders
    -> generated RNA and ATAC profiles
```

`X_multiflow` is a derived intermediate created by the encoder command. Users
should not construct it by hand.

## 1. Install

The paper workflow was run on Linux with a CUDA GPU. Clone MultiFlow and
install the optional paper dependencies:

```bash
git clone https://github.com/liuq-lab/MultiFlow.git
cd MultiFlow
python -m pip install --upgrade pip
python -m pip install -e ".[paper]"
```

MultiFlow uses the scDiffusion-X multimodal autoencoder. Clone the audited
revision beside the repository:

```bash
mkdir -p external
git clone https://github.com/EperLuo/scDiffusion-X.git external/scDiffusion-X
git -C external/scDiffusion-X checkout d60d928b635ab7a52ace030a641efd02137c193f
python -m pip install -e external/scDiffusion-X/scdiffusionX
```

The installed command is `multiflow`. The PyPI distribution is named
`multiflow-omics` because `multiflow` is occupied by an unrelated package.
Confirm that the paper workflow is available before starting a long run:

```bash
multiflow paper decode --help
```

If `paper` is not listed, update the checkout and reinstall it:

```bash
git pull
python -m pip install -e ".[paper]"
```

## 2. Initial H5MU input

The initial file contains paired raw profiles and biological labels:

```text
paired.h5mu
├── rna.X                    nonnegative integer RNA counts
├── atac.X                   binary ATAC accessibility
├── rna.obs["cell_type"]     cell-type name
└── matching RNA/ATAC obs_names in the same order
```

Perturbation data also require `rna.obs["perturbation"]`, with the control
condition named `control`. Audit any input before training:

```bash
multiflow data validate paired.h5mu
```

## 3. Task 1: cell-state generation

### 3.1 Download and split OpenProblem

Download the paired OpenProblem dataset used by scDiffusion-X. The file is
8.38 GB; the command verifies its size and MD5 before publishing it:

```bash
mkdir -p data
multiflow data download openproblem \
  --output data/openproblem_filtered.h5mu \
  --accept-license
```

Create the cell-type-stratified 80/20 split used by the benchmark. Split
before fitting either encoder so test cells do not determine encoder or latent
normalization parameters:

```bash
python workflows/prepare_generation_data.py \
  --input data/openproblem_filtered.h5mu \
  --output-dir data/openproblem \
  --train-fraction 0.8 \
  --seed 20260717
```

This writes `train.h5mu`, `test.h5mu`, `split_manifest.csv` and
`split_metadata.json`.

### 3.2 Train the RNA VAE

For generation, RNA counts are total-count normalized to 10,000 and `log1p`
transformed inside the trainer. The VAE has three 1,024-unit layers and a
128-dimensional L2-normalized latent state. Its decoder reconstructs
nonnegative normalized log-expression.

Download and extract the SCimilarity annotation weights used to initialize
the transferable hidden layers:

```bash
curl -L \
  "https://zenodo.org/records/8286452/files/annotation_model_v1.tar.gz?download=1" \
  -o external/annotation_model_v1.tar.gz
tar -xzf external/annotation_model_v1.tar.gz -C external
```

The directory must contain `encoder.ckpt`, `decoder.ckpt` and
`gene_order.tsv`. Fine-tune on training cells only:

```bash
python workflows/train_rna_vae.py \
  --train-h5mu data/openproblem/train.h5mu \
  --scimilarity-model-dir external/annotation_model_v1 \
  --output-dir runs/openproblem/rna_vae \
  --max-steps 200000 \
  --checkpoint-every 50000 \
  --batch-size 128 \
  --seed 1234 \
  --device cuda
```

The final checkpoint is
`runs/openproblem/rna_vae/rna_vae_step_199999.pt`. If an earlier checkpoint is
selected, use a validation subset of the training split, not the test cells.

### 3.3 Train the ATAC autoencoder

Generation uses the ATAC branch of a scDiffusion-X multimodal autoencoder.
RNA raw counts and binary ATAC are passed to this autoencoder; the ATAC branch
has a 128-dimensional state and a Bernoulli decoder.

```bash
python workflows/train_multimodal_ae.py \
  --scdiffusion-x-root external/scDiffusion-X \
  --train-h5mu data/openproblem/train.h5mu \
  --output-dir runs/openproblem/multimodal_ae \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --dataset-name openproblem \
  --epochs 300 \
  --seed 20260717
```

The stable checkpoint is
`runs/openproblem/multimodal_ae/checkpoints/final.ckpt`.

### 3.4 Encode the training profiles

This command performs both encoder calls. It applies
`normalize_total(1e4)+log1p` only to the RNA VAE input and passes binary ATAC
directly to the multimodal AE. Checkpoint hashes and encoder scale factors are
stored in the output H5MU.

```bash
multiflow paper encode \
  --task generation \
  --input data/openproblem/train.h5mu \
  --output data/openproblem/train_encoded.h5mu \
  --scdiffusion-x-root external/scDiffusion-X \
  --multimodal-ae-checkpoint runs/openproblem/multimodal_ae/checkpoints/final.ckpt \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --rna-vae-checkpoint runs/openproblem/rna_vae/rna_vae_step_199999.pt \
  --device cuda
```

Only this derived file contains `rna.obsm["X_multiflow"]` and
`atac.obsm["X_multiflow"]`.

### 3.5 Train MultiFlow

MultiFlow starts from one joint Gaussian random variable in the concatenated
RNA-ATAC latent space. Its RNA and ATAC coordinate blocks are marginally
independent, while the two flow branches exchange information through three
bidirectional cross-attention modules. The target velocity is the endpoint
difference along the linear path between each block of the joint Gaussian
source and the corresponding observed latent state; RNA and ATAC mean-squared
velocity losses are added.

```bash
multiflow train \
  --input data/openproblem/train_encoded.h5mu \
  --output runs/openproblem/multiflow_generation \
  --model cell-state \
  --condition-key cell_type \
  --epochs 600 \
  --batch-size 512 \
  --learning-rate 1e-4 \
  --hidden-dim 512 \
  --num-blocks 2 \
  --cross-attention-dim 64 \
  --sampling-steps 100 \
  --seed 0 \
  --device cuda \
  --hash-input
```

Latent means and standard deviations are estimated from training cells and
saved in `model.pt`. They are inverted automatically after sampling.

### 3.6 Generate and decode paired profiles

```bash
multiflow generate \
  --run runs/openproblem/multiflow_generation \
  --output runs/openproblem/generated_cd4_latent.h5mu \
  --cell-type "CD4+ T activated" \
  --n 1000 \
  --steps 100 \
  --batch-size 512 \
  --seed 0 \
  --device cuda

multiflow paper decode \
  --task generation \
  --input runs/openproblem/generated_cd4_latent.h5mu \
  --reference data/openproblem/train_encoded.h5mu \
  --output runs/openproblem/generated_cd4_profiles.h5mu \
  --scdiffusion-x-root external/scDiffusion-X \
  --multimodal-ae-checkpoint runs/openproblem/multimodal_ae/checkpoints/final.ckpt \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --rna-vae-checkpoint runs/openproblem/rna_vae/rna_vae_step_199999.pt \
  --seed 0 \
  --device cuda
```

Decoded RNA is a nonnegative normalized-log reconstruction. Decoded ATAC is a
Bernoulli sample and therefore contains only 0 and 1.

## 4. Task 2: perturbation prediction

### 4.1 Prepare the perturbation H5MU

The paper uses processed GSE274113 paired multiome data. Download the exact
version used by this tutorial from Zenodo. The file is 31.15 GB; the command
supports resuming and verifies its byte size and MD5 before publishing it at
the requested path:

```bash
multiflow data download gse274113 \
  --output data/GSE274113_filtered.h5mu \
  --accept-license

export PERTURBATION_H5MU=data/GSE274113_filtered.h5mu
multiflow data validate "$PERTURBATION_H5MU"
```

The fixed dataset version is
[Zenodo 10.5281/zenodo.21986866](https://doi.org/10.5281/zenodo.21986866);
the DOI representing all current and future versions is
[10.5281/zenodo.21986865](https://doi.org/10.5281/zenodo.21986865). The source
study is [GSE274113](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE274113).
Review the Zenodo record and source-data terms before passing
`--accept-license`.

### 4.2 Train and run the perturbation encoder

Perturbation prediction does **not** use the generation RNA VAE. Both RNA and
ATAC are encoded by the scDiffusion-X multimodal AE: RNA uses a
negative-binomial raw-count decoder and ATAC uses a Bernoulli decoder.

```bash
python workflows/train_multimodal_ae.py \
  --scdiffusion-x-root external/scDiffusion-X \
  --train-h5mu "$PERTURBATION_H5MU" \
  --output-dir runs/perturbation/multimodal_ae \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --dataset-name GSE274113 \
  --epochs 100 \
  --seed 20260717

multiflow paper encode \
  --task perturbation \
  --input "$PERTURBATION_H5MU" \
  --output data/GSE274113_encoded.h5mu \
  --scdiffusion-x-root external/scDiffusion-X \
  --multimodal-ae-checkpoint runs/perturbation/multimodal_ae/checkpoints/final.ckpt \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --device cuda
```

The original leave-one-cell-type-out notebook used one fixed multimodal AE and
then applied the holdout to flow training. If encoder fitting is part of the
evaluation split, create the raw fold first and retrain the AE within each
fold. Do not claim fold-level encoder isolation unless that run was performed.

### 4.3 Build one leave-one-cell-type-out fold

The notebook removes all non-control cells of the held-out type from flow
training; its control cells remain. Each context is the mean concatenated
RNA/ATAC encoder state over the remaining flow-training rows.

```bash
multiflow paper prepare-perturbation-fold \
  --input data/GSE274113_encoded.h5mu \
  --output data/GSE274113_CD4T_fold.h5mu \
  --held-out-cell-type "CD4+ T activated" \
  --condition-key cell_type \
  --perturbation-key perturbation \
  --control-perturbation control
```

### 4.4 Train, predict and decode

```bash
multiflow train \
  --input data/GSE274113_CD4T_fold.h5mu \
  --output runs/perturbation/CD4T \
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
  --seed 0 \
  --device cuda

multiflow generate \
  --run runs/perturbation/CD4T \
  --output runs/perturbation/CD4T_STAT1_latent_uncorrected.h5mu \
  --cell-type "CD4+ T activated" \
  --perturbation "STAT1" \
  --n 500 \
  --steps 50 \
  --batch-size 256 \
  --seed 0 \
  --device cuda

multiflow paper debias-perturbation \
  --input runs/perturbation/CD4T_STAT1_latent_uncorrected.h5mu \
  --training-fold data/GSE274113_CD4T_fold.h5mu \
  --output runs/perturbation/CD4T_STAT1_latent.h5mu \
  --held-out-cell-type "CD4+ T activated" \
  --perturbation "STAT1" \
  --condition-key cell_type \
  --perturbation-key perturbation \
  --control-perturbation control \
  --alpha 1

multiflow paper decode \
  --task perturbation \
  --input runs/perturbation/CD4T_STAT1_latent.h5mu \
  --reference data/GSE274113_CD4T_fold.h5mu \
  --output runs/perturbation/CD4T_STAT1_profiles.h5mu \
  --scdiffusion-x-root external/scDiffusion-X \
  --multimodal-ae-checkpoint runs/perturbation/multimodal_ae/checkpoints/final.ckpt \
  --encoder-config workflows/encoder_multimodal_128.yaml \
  --batch-size 2048 \
  --seed 0 \
  --device cuda
```

The control perturbation is index 0 and its embedding is fixed to zero. Before
decoding, the paper workflow applies a training-only latent mean shift. For
each modality, it estimates the mean perturbation effect from the flow-training
cells, adds that effect to the held-out cell type's control mean, and shifts
the generated group to this target mean. The default `--alpha 1` uses the full
estimated effect. No held-out perturbed cells are used. RNA is then decoded as
a raw-count expectation; ATAC is a binary Bernoulli output.

## 5. Paper settings

| Setting | Generation | Perturbation |
|---|---:|---:|
| RNA encoder | 128-d RNA VAE | 128-d multimodal AE RNA branch |
| RNA encoder input | normalize 1e4 + log1p | raw counts |
| ATAC encoder | 128-d multimodal AE ATAC branch | same |
| ATAC encoder input | binary | binary |
| Flow hidden width | 512 | 512 |
| Residual blocks per branch | 2 | 2 |
| Cross-attention feature width | 64 | 64 |
| Flow epochs | 1600 | 600 |
| Batch size | 512 | 512 |
| Adam learning rate | 1e-4 | 1e-4 |
| Midpoint ODE steps | 100 | 50 |
| Sampling batch size | 512 | 256 |
| Post-sampling latent mean shift | none | training-only, alpha = 1 |

At each integration step, RNA queries attend to ATAC keys and values, and
ATAC queries attend to RNA keys and values. The two vector fields are updated
inside one coupled system; the modalities are not generated independently.

## 6. Reproducibility checks

Keep the raw H5MU checksum, split manifest, exact feature order, matrix scales,
encoder configuration and checkpoint hashes, scDiffusion-X scale factors,
flow latent statistics, label mappings, training settings and sampling
settings with every result. `run.json` and H5MU metadata record these items.
Never exchange a decoder checkpoint after training the flow.

## 7. Advanced: latent-only API

The lower-level `multiflow train` and `multiflow generate` commands remain
useful when audited encoder latents already exist. Their input contract is in
[H5MU contract](h5mu_contract.md). This is an internal interface, not the
recommended starting point for a new user.
