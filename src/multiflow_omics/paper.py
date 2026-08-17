"""Bridges between raw paired H5MU profiles and the paper latent workflow.

The core MultiFlow model intentionally has no generic biological decoder.  The
paper workflow instead uses the exact task-specific encoders that created its
training latents and the matching decoders supplied by scDiffusion-X.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse

from .h5mu import prepare_h5mu_for_write, sha256_file
from .paper_encoders import load_rna_vae_checkpoint


def _paper_dependencies() -> tuple[Any, Any]:
    try:
        import anndata as ad
        import mudata as mu
    except ImportError as exc:
        raise RuntimeError(
            "The paper workflow needs AnnData and MuData. Install "
            "`python -m pip install -e '.[paper]'`."
        ) from exc
    return ad, mu


def _scanpy_dependency() -> Any:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "Generation RNA preprocessing needs Scanpy. Install "
            "`python -m pip install -e '.[paper]'`."
        ) from exc
    return sc


def _add_scdiffusionx(root: str | Path) -> Path:
    source = Path(root).expanduser().resolve()
    candidates = (source / "scdiffusionX" / "src", source / "src", source)
    for candidate in candidates:
        if (candidate / "scdiffusionX").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(
        f"Could not find scdiffusionX below {source}; tried "
        + ", ".join(map(str, candidates))
    )


def _dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    result = np.asarray(matrix, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("expected a finite two-dimensional matrix")
    return result


def _read_checkpoint(path: str | Path, *, device: str) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("scDiffusion-X checkpoint is missing state_dict")
    return payload


@torch.no_grad()
def _encode_rna_vae(
    adata: Any,
    checkpoint: str | Path,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    sc = _scanpy_dependency()
    normalized = adata.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    model = load_rna_vae_checkpoint(
        checkpoint,
        n_genes=normalized.n_vars,
        device=device,
    )
    chunks: list[np.ndarray] = []
    for start in range(0, normalized.n_obs, batch_size):
        stop = min(start + batch_size, normalized.n_obs)
        x = torch.as_tensor(
            _dense_float32(normalized.X[start:stop]),
            dtype=torch.float32,
            device=device,
        )
        chunks.append(model(x, return_latent=True).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def encode_paper_h5mu(
    *,
    input_path: str | Path,
    output_path: str | Path,
    task: str,
    scdiffusionx_root: str | Path,
    multimodal_ae_checkpoint: str | Path,
    encoder_config: str | Path,
    rna_vae_checkpoint: str | Path | None,
    condition_key: str,
    device: str,
    batch_size: int,
) -> Path:
    """Encode raw paired profiles using the task-specific paper encoders."""
    _, mu = _paper_dependencies()
    if task not in {"generation", "perturbation"}:
        raise ValueError("task must be generation or perturbation")
    if task == "generation" and rna_vae_checkpoint is None:
        raise ValueError("generation encoding requires an RNA VAE checkpoint")

    _add_scdiffusionx(scdiffusionx_root)
    from scdiffusionX.DiffusionBackbone.multimodal_datasets import (
        MultimodalDataset_cell,
    )

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    ae_checkpoint = Path(multimodal_ae_checkpoint).expanduser().resolve()
    config = Path(encoder_config).expanduser().resolve()
    for path in (source, ae_checkpoint, config):
        if not path.is_file():
            raise FileNotFoundError(path)

    mdata = mu.read_h5mu(source)
    if set(("rna", "atac")).difference(mdata.mod):
        raise ValueError("input H5MU must contain rna and atac modalities")
    if not np.array_equal(
        np.asarray(mdata["rna"].obs_names.astype(str)),
        np.asarray(mdata["atac"].obs_names.astype(str)),
    ):
        raise ValueError("RNA and ATAC cell order must match")
    if condition_key not in mdata["rna"].obs:
        raise KeyError(f"rna.obs[{condition_key!r}] is missing")

    dataset = MultimodalDataset_cell(
        data_path=str(source),
        ae_path=str(ae_checkpoint),
        condition=condition_key,
        encoder_config=str(config),
        dev=device,
    )
    mm_rna = np.asarray(dataset.adata_rna, dtype=np.float32).squeeze(1)
    atac = np.asarray(dataset.adata_atac, dtype=np.float32).squeeze(1)
    if task == "generation":
        rna = _encode_rna_vae(
            mdata["rna"],
            Path(rna_vae_checkpoint),
            device=device,
            batch_size=batch_size,
        )
        rna_encoder = "scDiffusion RNA VAE; normalize_total(1e4)+log1p"
    else:
        rna = mm_rna
        rna_encoder = "scDiffusion-X multimodal AE RNA branch; raw counts"
    if rna.shape[0] != mdata.n_obs or atac.shape[0] != mdata.n_obs:
        raise ValueError("encoder output rows do not match the paired input cells")

    mdata["rna"].obsm["X_multiflow"] = rna
    mdata["atac"].obsm["X_multiflow"] = atac
    metadata = {
        "schema_version": 1,
        "task": task,
        "rna_encoder": rna_encoder,
        "atac_encoder": "scDiffusion-X multimodal AE ATAC branch; binary input",
        "rna_latent_dim": int(rna.shape[1]),
        "atac_latent_dim": int(atac.shape[1]),
        "rna_std10": float(torch.as_tensor(dataset.rna_std10).cpu().item()),
        "atac_std10": float(torch.as_tensor(dataset.atac_std10).cpu().item()),
        "multimodal_ae_sha256": sha256_file(ae_checkpoint),
        "encoder_config_sha256": sha256_file(config),
        "rna_vae_sha256": (
            sha256_file(Path(rna_vae_checkpoint)) if rna_vae_checkpoint else None
        ),
        "condition_key": condition_key,
        "source_h5mu": str(source),
    }
    mdata.uns["multiflow_encoder"] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(mdata).write_h5mu(output)
    return output


def prepare_perturbation_fold(
    *,
    input_path: str | Path,
    output_path: str | Path,
    held_out_cell_type: str,
    condition_key: str,
    perturbation_key: str,
    control_perturbation: str,
) -> Path:
    """Create the exact flow-level LOCO split used by the perturbation notebook."""
    _, mu = _paper_dependencies()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    mdata = mu.read_h5mu(source)
    obs = mdata["rna"].obs
    for key in (condition_key, perturbation_key):
        if key not in obs:
            raise KeyError(f"rna.obs[{key!r}] is missing")
    for modality in ("rna", "atac"):
        if "X_multiflow" not in mdata[modality].obsm:
            raise KeyError(f"{modality}.obsm['X_multiflow'] is missing")

    cell_type = obs[condition_key].astype(str).to_numpy()
    perturbation = obs[perturbation_key].astype(str).to_numpy()
    if held_out_cell_type not in set(cell_type):
        raise ValueError(f"unknown held-out cell type: {held_out_cell_type!r}")
    holdout = (cell_type == held_out_cell_type) & (
        perturbation != control_perturbation
    )
    if not holdout.any():
        raise ValueError("the requested fold contains no held-out perturbed cells")
    train_names = np.asarray(mdata["rna"].obs_names.astype(str))[~holdout]
    train = mdata[train_names, :].copy()

    rna = _dense_float32(train["rna"].obsm["X_multiflow"])
    atac = _dense_float32(train["atac"].obsm["X_multiflow"])
    labels = train["rna"].obs[condition_key].astype(str).to_numpy()
    classes = sorted(set(labels.tolist()))
    context = np.stack(
        [np.concatenate((rna[labels == name], atac[labels == name]), axis=1).mean(0)
         for name in classes],
        axis=0,
    ).astype(np.float32)
    train.uns["multiflow_context_matrix"] = context
    train.uns["multiflow_context_classes"] = np.asarray(classes, dtype=object)
    train.uns["multiflow_fold"] = json.dumps(
        {
            "schema_version": 1,
            "protocol": "leave-one-cell-type-out_noncontrol",
            "held_out_cell_type": held_out_cell_type,
            "control_perturbation": control_perturbation,
            "n_input": int(mdata.n_obs),
            "n_train": int(train.n_obs),
            "n_held_out_perturbed": int(holdout.sum()),
            "context_source": "unstandardized paired encoder latents from flow training rows",
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(train).write_h5mu(output)
    return output


def debias_perturbation_h5mu(
    *,
    generated_path: str | Path,
    training_fold_path: str | Path,
    output_path: str | Path,
    held_out_cell_type: str,
    perturbation: str,
    condition_key: str,
    perturbation_key: str,
    control_perturbation: str,
    alpha: float = 1.0,
    min_cells: int = 5,
) -> Path:
    """Apply the executed training-only perturbation mean-shift correction."""
    _, mu = _paper_dependencies()
    generated_file = Path(generated_path).expanduser().resolve()
    fold_file = Path(training_fold_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    generated = mu.read_h5mu(generated_file)
    fold = mu.read_h5mu(fold_file)
    obs = fold["rna"].obs
    cell_types = obs[condition_key].astype(str).to_numpy()
    perturbations = obs[perturbation_key].astype(str).to_numpy()
    control_mask = perturbations == control_perturbation
    target_mask = perturbations == perturbation
    heldout_control_mask = control_mask & (cell_types == held_out_cell_type)
    for name, mask in (
        ("training controls", control_mask),
        (f"training perturbation {perturbation!r}", target_mask),
        (f"held-out controls for {held_out_cell_type!r}", heldout_control_mask),
    ):
        if int(mask.sum()) < min_cells:
            raise ValueError(f"{name} has {int(mask.sum())} cells; need at least {min_cells}")

    for modality in ("rna", "atac"):
        train_latent = _dense_float32(fold[modality].obsm["X_multiflow"])
        generated_latent = _dense_float32(generated[modality].X)
        training_delta = (
            train_latent[target_mask].mean(axis=0)
            - train_latent[control_mask].mean(axis=0)
        )
        target_mean = (
            train_latent[heldout_control_mask].mean(axis=0)
            + float(alpha) * training_delta
        )
        corrected = generated_latent + (
            target_mean - generated_latent.mean(axis=0)
        )
        generated[modality].X = corrected.astype(np.float32, copy=False)

    generated.uns["multiflow_debias"] = json.dumps(
        {
            "schema_version": 1,
            "method": "training_only_perturbation_mean_shift",
            "held_out_cell_type": held_out_cell_type,
            "perturbation": perturbation,
            "control_perturbation": control_perturbation,
            "alpha": float(alpha),
            "training_fold": str(fold_file),
            "n_training_controls": int(control_mask.sum()),
            "n_training_perturbation": int(target_mask.sum()),
            "n_heldout_controls": int(heldout_control_mask.sum()),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(generated).write_h5mu(output)
    return output


def _encoder_metadata(mdata: Any) -> dict[str, Any]:
    raw = mdata.uns.get("multiflow_encoder")
    if raw is None:
        raise ValueError("reference H5MU has no multiflow_encoder metadata")
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _load_multimodal_decoder(
    *,
    scdiffusionx_root: str | Path,
    reference_path: Path,
    checkpoint: Path,
    encoder_config: Path,
    condition_key: str,
    device: str,
) -> tuple[Any, Any]:
    _, mu = _paper_dependencies()
    _add_scdiffusionx(scdiffusionx_root)
    import yaml
    from scdiffusionX.Autoencoder.models.base.encoder_model import EncoderModel

    reference = mu.read_h5mu(reference_path)
    config = yaml.safe_load(encoder_config.read_text(encoding="utf-8"))
    config["encoder_kwargs"]["rna"]["norm_type"] = "batchnorm"
    config["encoder_kwargs"]["atac"]["norm_type"] = "batchnorm"
    model = EncoderModel(
        in_dim={"rna": reference["rna"].n_vars, "atac": reference["atac"].n_vars},
        n_cat=reference["rna"].obs[condition_key].astype(str).nunique(),
        conditioning_covariate=condition_key,
        encoder_type="learnt_autoencoder",
        **config,
    )
    model.load_state_dict(_read_checkpoint(checkpoint, device=device)["state_dict"])
    model.to(device).eval()
    return model, reference


def _log_library_stats_by_class(
    rna: Any,
    *,
    condition_key: str,
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Match scDiffusion-X RNA library statistics without densifying profiles."""
    labels = rna.obs[condition_key].astype(str).to_numpy()
    matrix = rna.X
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).reshape(-1)
    if row_sums.shape[0] != rna.n_obs or not np.isfinite(row_sums).all():
        raise ValueError("reference RNA library sizes are invalid")
    if np.any(row_sums <= 0):
        raise ValueError("reference RNA contains cells with zero library size")
    log_sizes = np.log(row_sums)
    means: list[float] = []
    standard_deviations: list[float] = []
    for name in classes:
        values = log_sizes[labels == name]
        if values.size == 0:
            raise ValueError(f"reference has no RNA cells for condition {name!r}")
        means.append(float(values.mean()))
        # torch.std, used by scDiffusion-X, applies Bessel's correction.
        standard_deviations.append(
            float(values.std(ddof=1)) if values.size > 1 else float("nan")
        )
    return (
        np.asarray(means, dtype=np.float32),
        np.asarray(standard_deviations, dtype=np.float32),
    )


@torch.no_grad()
def decode_paper_h5mu(
    *,
    generated_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    task: str,
    scdiffusionx_root: str | Path,
    multimodal_ae_checkpoint: str | Path,
    encoder_config: str | Path,
    rna_vae_checkpoint: str | Path | None,
    condition_key: str,
    device: str,
    batch_size: int,
    seed: int,
) -> Path:
    """Decode a generated latent H5MU with the matching paper decoders."""
    ad, mu = _paper_dependencies()
    if task not in {"generation", "perturbation"}:
        raise ValueError("task must be generation or perturbation")
    generated_file = Path(generated_path).expanduser().resolve()
    reference_file = Path(reference_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    checkpoint = Path(multimodal_ae_checkpoint).expanduser().resolve()
    config = Path(encoder_config).expanduser().resolve()
    generated = mu.read_h5mu(generated_file)
    rna_latent = _dense_float32(generated["rna"].X)
    atac_latent = _dense_float32(generated["atac"].X)
    model, reference = _load_multimodal_decoder(
        scdiffusionx_root=scdiffusionx_root,
        reference_path=reference_file,
        checkpoint=checkpoint,
        encoder_config=config,
        condition_key=condition_key,
        device=device,
    )
    metadata = _encoder_metadata(reference)
    rna_std10 = float(metadata["rna_std10"])
    atac_std10 = float(metadata["atac_std10"])
    classes = sorted(reference["rna"].obs[condition_key].astype(str).unique())
    generated_labels = generated["rna"].obs[condition_key].astype(str).to_numpy()
    mapping = {name: index for index, name in enumerate(classes)}
    try:
        labels = np.asarray([mapping[name] for name in generated_labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"generated cell type is absent from reference: {exc.args[0]}") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rna_blocks: list[np.ndarray] = []
    atac_blocks: list[np.ndarray] = []
    rna_vae = None
    library_mean: np.ndarray | None = None
    library_sd: np.ndarray | None = None
    if task == "generation":
        if rna_vae_checkpoint is None:
            raise ValueError("generation decoding requires an RNA VAE checkpoint")
        rna_vae = load_rna_vae_checkpoint(
            rna_vae_checkpoint,
            n_genes=reference["rna"].n_vars,
            device=device,
        )
    else:
        library_mean, library_sd = _log_library_stats_by_class(
            reference["rna"],
            condition_key=condition_key,
            classes=classes,
        )

    from torch.distributions import Bernoulli, Normal
    from torch.nn import functional as F

    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        z_atac = torch.as_tensor(
            atac_latent[start:stop] * atac_std10,
            dtype=torch.float32,
            device=device,
        )
        atac_probability = torch.sigmoid(model.decoder["atac"](z_atac))
        atac_blocks.append(Bernoulli(probs=atac_probability).sample().cpu().numpy())

        z_rna = torch.as_tensor(rna_latent[start:stop], dtype=torch.float32, device=device)
        if task == "generation":
            rna_out = rna_vae(z_rna, return_decoded=True)
        else:
            z_rna = z_rna * rna_std10
            logits = model.decoder["rna"](z_rna)
            batch_labels = labels[start:stop]
            assert library_mean is not None and library_sd is not None
            mean = torch.as_tensor(library_mean)[batch_labels].to(device)
            sd = torch.nan_to_num(
                torch.as_tensor(library_sd)[batch_labels].to(device),
                nan=1e-4,
                posinf=1e-4,
                neginf=1e-4,
            ).clamp_min(1e-4)
            library_size = torch.exp(Normal(mean, sd).sample()).view(-1, 1)
            rna_out = F.softmax(logits, dim=1) * library_size
        rna_blocks.append(rna_out.cpu().numpy())

    obs = generated["rna"].obs.copy()
    rna = ad.AnnData(
        X=np.concatenate(rna_blocks).astype(np.float32, copy=False),
        obs=obs.copy(),
        var=reference["rna"].var.copy(),
    )
    atac = ad.AnnData(
        X=np.concatenate(atac_blocks).astype(np.float32, copy=False),
        obs=obs.copy(),
        var=reference["atac"].var.copy(),
    )
    rna.uns["matrix_scale"] = (
        "log1p_normalized_1e4_reconstruction"
        if task == "generation"
        else "raw_count_expectation"
    )
    atac.uns["matrix_scale"] = "binary"
    decoded = mu.MuData({"rna": rna, "atac": atac})
    decoded.uns["multiflow_decode"] = json.dumps(
        {
            "schema_version": 1,
            "task": task,
            "source_generated_h5mu": str(generated_file),
            "reference_h5mu": str(reference_file),
            "multimodal_ae_sha256": sha256_file(checkpoint),
            "rna_vae_sha256": (
                sha256_file(Path(rna_vae_checkpoint)) if rna_vae_checkpoint else None
            ),
            "seed": int(seed),
        },
        sort_keys=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_h5mu_for_write(decoded).write_h5mu(output)
    return output
