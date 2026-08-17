#!/usr/bin/env python3
"""Create the paper's fixed 80/20 cell-type-stratified H5MU split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mudata as mu
import numpy as np
import pandas as pd

from multiflow_omics.h5mu import prepare_h5mu_for_write, validate_h5mu


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split(labels: pd.Series, fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[tuple[str, str, str]] = []
    for name in sorted(labels.unique()):
        cells = labels.index[labels == name].to_numpy(dtype=str)
        if len(cells) < 2:
            raise ValueError(f"cell type {name!r} needs at least two cells")
        cells = cells[rng.permutation(len(cells))]
        n_train = min(max(int(np.floor(len(cells) * fraction)), 1), len(cells) - 1)
        records.extend((cell, name, "train") for cell in cells[:n_train])
        records.extend((cell, name, "test") for cell in cells[n_train:])
    return pd.DataFrame(records, columns=["obs_name", "cell_type", "split"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition-key", default="cell_type")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    report = validate_h5mu(source, condition_key=args.condition_key)
    if not report["paired_raw_contract_ok"]:
        raise ValueError("input RNA must be raw counts and input ATAC must be binary")
    mdata = mu.read_h5mu(source)
    labels = mdata["rna"].obs[args.condition_key].astype(str).copy()
    labels.index = mdata["rna"].obs_names.astype(str)
    manifest = _split(labels, args.train_fraction, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "split_manifest.csv", index=False)

    for split_name in ("train", "test"):
        names = manifest.loc[manifest["split"] == split_name, "obs_name"].tolist()
        subset = mdata[names, :].copy()
        destination = output / f"{split_name}.h5mu"
        prepare_h5mu_for_write(subset).write_h5mu(destination)
        print(f"saved {destination} ({subset.n_obs} paired cells)")

    metadata = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": _sha256(source),
        "train_fraction": float(args.train_fraction),
        "seed": int(args.seed),
        "condition_key": args.condition_key,
        "n_total": int(mdata.n_obs),
        "n_train": int((manifest["split"] == "train").sum()),
        "n_test": int((manifest["split"] == "test").sum()),
    }
    (output / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

