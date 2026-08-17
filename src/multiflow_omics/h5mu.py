"""Audited H5MU input and output helpers for MultiFlow.

MultiFlow learns in latent space.  H5MU is used as the paired container so
cell identifiers, biological conditions, and representation provenance stay
together instead of being split across anonymous NPZ arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

DEFAULT_RNA_REPRESENTATION = "X_multiflow"
DEFAULT_ATAC_REPRESENTATION = "X_multiflow"


@contextmanager
def _quiet_mudata_update_warnings() -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"From 0\.4 \.update\(\).*",
            category=FutureWarning,
        )
        yield


def _require_h5mu_dependencies() -> tuple[Any, Any]:
    try:
        import anndata as ad
        import mudata as mu
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "H5MU support requires mudata and anndata. Install the package with "
            "`python -m pip install multiflow-omics`."
        ) from exc
    return ad, mu


def prepare_h5mu_for_write(mdata: Any) -> Any:
    """Coerce string indices to the portable HDF5 representation.

    Pandas 2/3 can infer nullable string dtypes for indices created while
    MuData aggregates modality annotations.  AnnData deliberately refuses to
    write that experimental dtype by default because older readers cannot
    consume it.  Storing the same values as ordinary Python strings keeps toy
    files and generated outputs readable across supported AnnData versions.
    """
    frames = [mdata.obs, mdata.var]
    for adata in mdata.mod.values():
        frames.extend([adata.obs, adata.var])
    for frame in frames:
        frame.index = pd.Index(
            np.asarray(frame.index.astype(str), dtype=object), dtype=object
        )
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.StringDtype):
                frame[column] = frame[column].astype(object)
    return mdata


def _to_memory(matrix: Any) -> Any:
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return matrix


def _dense_float32(matrix: Any, *, name: str) -> np.ndarray:
    matrix = _to_memory(matrix)
    if sparse.issparse(matrix):
        array = matrix.toarray()
    else:
        array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _representation(adata: Any, key: str, *, modality: str) -> np.ndarray:
    if key == "X":
        matrix = adata.X
    elif key in adata.obsm:
        matrix = adata.obsm[key]
    else:
        available = sorted(map(str, adata.obsm.keys()))
        raise ValueError(
            f"{modality}.obsm[{key!r}] is missing. Available obsm keys: {available}. "
            "Raw X is not silently treated as a MultiFlow latent representation."
        )
    return _dense_float32(matrix, name=f"{modality} representation {key!r}")


def _condition_values(mdata: Any, key: str) -> np.ndarray:
    candidates: list[tuple[str, Any]] = []
    if key in mdata.obs:
        candidates.append(("mdata.obs", mdata.obs[key]))
    for modality in ("rna", "atac"):
        if key in mdata.mod[modality].obs:
            candidates.append((f"{modality}.obs", mdata.mod[modality].obs[key]))
    if not candidates:
        raise ValueError(
            f"condition column {key!r} is absent from mdata.obs, rna.obs, and atac.obs"
        )
    reference = np.asarray(candidates[0][1].astype("string"), dtype=object)
    if pd.isna(reference).any():
        raise ValueError(f"condition column {key!r} contains missing values")
    for location, values in candidates[1:]:
        observed = np.asarray(values.astype("string"), dtype=object)
        if not np.array_equal(reference, observed):
            raise ValueError(
                f"condition column {key!r} differs between paired tables "
                f"({candidates[0][0]} versus {location})"
            )
    return reference.astype(str)


def encode_categories(
    values: np.ndarray, *, first_class: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """Encode categories deterministically, optionally reserving index zero."""
    classes = sorted(set(map(str, values.tolist())))
    if first_class is not None:
        if first_class not in classes:
            raise ValueError(
                f"required class {first_class!r} is absent; observed classes: {classes}"
            )
        classes.remove(first_class)
        classes.insert(0, first_class)
    mapping = {name: index for index, name in enumerate(classes)}
    codes = np.fromiter((mapping[str(value)] for value in values), dtype=np.int64)
    return codes, classes


def _open_h5mu(path: str | os.PathLike[str], *, backed: bool = False) -> Any:
    _, mu = _require_h5mu_dependencies()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with _quiet_mudata_update_warnings():
        return mu.read_h5mu(source, backed="r" if backed else None)


def _check_pairing(mdata: Any) -> None:
    missing = sorted({"rna", "atac"}.difference(mdata.mod))
    if missing:
        raise ValueError(f"H5MU is missing required modalities: {missing}")
    rna_names = np.asarray(mdata.mod["rna"].obs_names.astype(str))
    atac_names = np.asarray(mdata.mod["atac"].obs_names.astype(str))
    if not np.array_equal(rna_names, atac_names):
        raise ValueError("rna.obs_names and atac.obs_names must match in exactly the same order")
    if len(set(rna_names.tolist())) != len(rna_names):
        raise ValueError("paired cell identifiers must be unique")


@dataclass(frozen=True)
class PairedLatentData:
    """Paired latent arrays plus reproducible biological label mappings."""

    rna: np.ndarray
    atac: np.ndarray
    obs_names: np.ndarray
    labels: np.ndarray | None
    label_classes: list[str]
    perturbations: np.ndarray | None
    perturbation_classes: list[str]
    context_matrix: np.ndarray | None


def read_paired_latents(
    path: str | os.PathLike[str],
    *,
    rna_representation: str = DEFAULT_RNA_REPRESENTATION,
    atac_representation: str = DEFAULT_ATAC_REPRESENTATION,
    condition_key: str | None = "cell_type",
    perturbation_key: str | None = None,
    context_matrix_key: str = "multiflow_context_matrix",
    context_classes_key: str = "multiflow_context_classes",
    control_perturbation: str = "control",
) -> PairedLatentData:
    """Read paired latent representations and conditions from one H5MU file."""
    mdata = _open_h5mu(path)
    _check_pairing(mdata)
    rna = _representation(mdata.mod["rna"], rna_representation, modality="rna")
    atac = _representation(mdata.mod["atac"], atac_representation, modality="atac")
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("RNA and ATAC representations must contain the same paired rows")

    labels: np.ndarray | None = None
    label_classes: list[str] = []
    if condition_key is not None:
        labels, label_classes = encode_categories(_condition_values(mdata, condition_key))

    perturbations: np.ndarray | None = None
    perturbation_classes: list[str] = []
    context_matrix: np.ndarray | None = None
    if perturbation_key is not None:
        perturbations, perturbation_classes = encode_categories(
            _condition_values(mdata, perturbation_key),
            first_class=control_perturbation,
        )
        if context_matrix_key not in mdata.uns:
            raise ValueError(
                f"perturbation training requires mdata.uns[{context_matrix_key!r}]"
            )
        context_matrix = _dense_float32(
            mdata.uns[context_matrix_key], name=f"mdata.uns[{context_matrix_key!r}]"
        )
        if labels is None or context_matrix.shape[0] != len(label_classes):
            raise ValueError("context matrix rows must match the fixed cell-type class order")
        if context_classes_key not in mdata.uns:
            raise ValueError(
                f"perturbation training requires mdata.uns[{context_classes_key!r}] "
                "to verify context-matrix row order"
            )
        context_classes = list(map(str, np.asarray(mdata.uns[context_classes_key]).tolist()))
        if context_classes != label_classes:
            raise ValueError(
                "context-matrix class order differs from the fixed cell-type class order: "
                f"expected {label_classes}, observed {context_classes}"
            )

    return PairedLatentData(
        rna=rna,
        atac=atac,
        obs_names=np.asarray(mdata.mod["rna"].obs_names.astype(str)),
        labels=labels,
        label_classes=label_classes,
        perturbations=perturbations,
        perturbation_classes=perturbation_classes,
        context_matrix=context_matrix,
    )


def _matrix_blocks(matrix: Any, *, rows_per_block: int = 512) -> Any:
    """Yield every matrix row in bounded-memory blocks.

    ``np.arange`` is intentional: it works across in-memory arrays, scipy
    sparse matrices, and the backed sparse datasets used by supported AnnData
    releases.
    """
    n_rows = int(matrix.shape[0])
    for start in range(0, n_rows, rows_per_block):
        stop = min(start + rows_per_block, n_rows)
        row_index = np.arange(start, stop, dtype=np.intp)
        yield _to_memory(matrix[row_index, :])


def _matrix_audit(matrix: Any, *, expected: str) -> dict[str, Any]:
    """Audit a complete matrix without materializing it all at once."""
    n_rows, n_columns = map(int, matrix.shape)
    nonempty = n_rows > 0 and n_columns > 0
    finite = nonempty
    nonnegative = nonempty
    integer_like = nonempty
    binary = nonempty
    minimum = np.inf
    maximum = -np.inf
    observed_values = 0
    implicit_zeros = 0

    for block in _matrix_blocks(matrix):
        if sparse.issparse(block):
            values = np.asarray(block.data)
            implicit_zeros += int(np.prod(block.shape)) - int(block.nnz)
        else:
            values = np.asarray(block).reshape(-1)
        if values.size == 0:
            continue
        observed_values += int(values.size)
        block_finite = bool(np.isfinite(values).all())
        finite = finite and block_finite
        if not block_finite:
            nonnegative = integer_like = binary = False
            continue
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
        block_nonnegative = bool((values >= 0).all())
        block_integer = bool(np.allclose(values, np.round(values), atol=1e-6))
        nonnegative = nonnegative and block_nonnegative
        integer_like = integer_like and block_integer
        binary = binary and block_nonnegative and block_integer and bool((values <= 1).all())

    if implicit_zeros:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    if not observed_values and not implicit_zeros:
        finite = nonnegative = integer_like = binary = False
        minimum = maximum = np.nan

    if expected == "raw_counts":
        contract_ok = nonempty and finite and nonnegative and integer_like
    elif expected == "binary":
        contract_ok = nonempty and finite and binary
    elif expected == "latent":
        contract_ok = nonempty and finite
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown matrix contract: {expected}")
    return {
        "shape": [n_rows, n_columns],
        "finite": bool(finite),
        "nonnegative": bool(nonnegative),
        "integer_like": bool(integer_like),
        "binary": bool(binary),
        "min": None if not np.isfinite(minimum) else float(minimum),
        "max": None if not np.isfinite(maximum) else float(maximum),
        "expected_scale": expected,
        "audit_mode": "full_chunked",
        "contract_ok": bool(contract_ok),
    }


def _close_backed_h5mu(mdata: Any) -> None:
    file_manager = getattr(mdata, "file", None)
    close = getattr(file_manager, "close", None)
    if callable(close):
        close()


def validate_h5mu(
    path: str | os.PathLike[str],
    *,
    condition_key: str = "cell_type",
    rna_representation: str = DEFAULT_RNA_REPRESENTATION,
    atac_representation: str = DEFAULT_ATAC_REPRESENTATION,
) -> dict[str, Any]:
    """Validate pairing, raw scales, and availability of MultiFlow latents."""
    source = Path(path).expanduser().resolve()
    mdata = _open_h5mu(source, backed=True)
    try:
        _check_pairing(mdata)
        conditions = _condition_values(mdata, condition_key)
        rna_present = bool(
            rna_representation == "X" or rna_representation in mdata.mod["rna"].obsm
        )
        atac_present = bool(
            atac_representation == "X" or atac_representation in mdata.mod["atac"].obsm
        )
        report: dict[str, Any] = {
            "path": str(source),
            "n_obs": int(mdata.mod["rna"].n_obs),
            "condition_key": condition_key,
            "n_conditions": int(len(set(conditions.tolist()))),
            "rna": _matrix_audit(mdata.mod["rna"].X, expected="raw_counts"),
            "atac": _matrix_audit(mdata.mod["atac"].X, expected="binary"),
            "rna_representation": rna_representation,
            "atac_representation": atac_representation,
            "rna_representation_present": rna_present,
            "atac_representation_present": atac_present,
        }
        report["paired_raw_contract_ok"] = bool(
            report["rna"]["contract_ok"] and report["atac"]["contract_ok"]
        )
        if rna_present:
            rna_matrix = (
                mdata.mod["rna"].X
                if rna_representation == "X"
                else mdata.mod["rna"].obsm[rna_representation]
            )
            report["rna_latent"] = _matrix_audit(rna_matrix, expected="latent")
        if atac_present:
            atac_matrix = (
                mdata.mod["atac"].X
                if atac_representation == "X"
                else mdata.mod["atac"].obsm[atac_representation]
            )
            report["atac_latent"] = _matrix_audit(atac_matrix, expected="latent")
        report["latent_ready"] = bool(
            rna_present
            and atac_present
            and report["rna_latent"]["contract_ok"]
            and report["atac_latent"]["contract_ok"]
            and report["rna_latent"]["shape"][0] == report["n_obs"]
            and report["atac_latent"]["shape"][0] == report["n_obs"]
        )
        return report
    finally:
        _close_backed_h5mu(mdata)


def write_generated_h5mu(
    path: str | os.PathLike[str],
    rna_latent: np.ndarray,
    atac_latent: np.ndarray,
    *,
    cell_type: str | None,
    cell_type_code: int | None,
    perturbation: str | None = None,
    perturbation_code: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write paired generated latent states as a self-describing H5MU file."""
    ad, mu = _require_h5mu_dependencies()
    rna = _dense_float32(rna_latent, name="generated RNA latent")
    atac = _dense_float32(atac_latent, name="generated ATAC latent")
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("generated RNA and ATAC must have the same number of cells")
    obs_names = pd.Index(
        np.asarray([f"multiflow_{index:08d}" for index in range(rna.shape[0])], dtype=object),
        dtype=object,
    )
    obs = pd.DataFrame(index=obs_names)
    if cell_type is not None:
        obs["cell_type"] = np.asarray([cell_type] * rna.shape[0], dtype=object)
        obs["cell_type_code"] = int(cell_type_code)
    if perturbation is not None:
        obs["perturbation"] = np.asarray([perturbation] * rna.shape[0], dtype=object)
        obs["perturbation_code"] = int(perturbation_code)
    rna_adata = ad.AnnData(
        X=rna,
        obs=obs.copy(),
        var=pd.DataFrame(
            index=pd.Index(
                np.asarray(
                    [f"rna_latent_{index:04d}" for index in range(rna.shape[1])],
                    dtype=object,
                ),
                dtype=object,
            )
        ),
    )
    atac_adata = ad.AnnData(
        X=atac,
        obs=obs.copy(),
        var=pd.DataFrame(
            index=pd.Index(
                np.asarray(
                    [f"atac_latent_{index:04d}" for index in range(atac.shape[1])],
                    dtype=object,
                ),
                dtype=object,
            )
        ),
    )
    for adata in (rna_adata, atac_adata):
        adata.uns["matrix_scale"] = "latent"
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".h5mu":
        raise ValueError("generated output must use the .h5mu suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _quiet_mudata_update_warnings():
        mdata = mu.MuData({"rna": rna_adata, "atac": atac_adata})
        safe_metadata = json.loads(json.dumps(metadata or {}, sort_keys=True))
        mdata.uns["multiflow"] = safe_metadata
        prepare_h5mu_for_write(mdata)
        mdata.write_h5mu(destination)
    return destination


def write_toy_h5mu(path: str | os.PathLike[str], *, seed: int = 7) -> Path:
    """Create a small synthetic, paired H5MU file for tutorials and smoke tests."""
    ad, mu = _require_h5mu_dependencies()
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_peaks, latent_dim = 48, 12, 20, 4
    cell_types = np.asarray(["B cell", "T cell", "Monocyte"] * 16, dtype=object)
    obs_names = pd.Index(
        np.asarray([f"toy_cell_{index:03d}" for index in range(n_cells)], dtype=object),
        dtype=object,
    )
    obs = pd.DataFrame({"cell_type": cell_types}, index=obs_names)
    rna_adata = ad.AnnData(
        X=sparse.csr_matrix(
            rng.poisson(1.2, size=(n_cells, n_genes)).astype(np.float32)
        ),
        obs=obs.copy(),
        var=pd.DataFrame(
            index=pd.Index(
                np.asarray(
                    [f"gene_{index:03d}" for index in range(n_genes)], dtype=object
                ),
                dtype=object,
            )
        ),
    )
    atac_adata = ad.AnnData(
        X=sparse.csr_matrix(
            rng.binomial(1, 0.16, size=(n_cells, n_peaks)).astype(np.float32)
        ),
        obs=obs.copy(),
        var=pd.DataFrame(
            index=pd.Index(
                np.asarray(
                    [f"peak_{index:03d}" for index in range(n_peaks)], dtype=object
                ),
                dtype=object,
            )
        ),
    )
    offsets = {"B cell": -1.0, "T cell": 0.0, "Monocyte": 1.0}
    signal = np.asarray([offsets[str(value)] for value in cell_types], dtype=np.float32)[
        :, None
    ]
    rna_adata.obsm[DEFAULT_RNA_REPRESENTATION] = (
        signal + rng.normal(0, 0.35, (n_cells, latent_dim))
    ).astype(np.float32)
    atac_adata.obsm[DEFAULT_ATAC_REPRESENTATION] = (
        signal + rng.normal(0, 0.35, (n_cells, latent_dim))
    ).astype(np.float32)
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".h5mu":
        raise ValueError("example output must use the .h5mu suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _quiet_mudata_update_warnings():
        mdata = mu.MuData({"rna": rna_adata, "atac": atac_adata})
        mdata.uns["multiflow_example"] = {
            "synthetic": True,
            "raw_rna_scale": "raw_counts",
            "raw_atac_scale": "binary",
            "latent_representation": DEFAULT_RNA_REPRESENTATION,
            "seed": int(seed),
        }
        prepare_h5mu_for_write(mdata)
        mdata.write_h5mu(destination)
    return destination


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
