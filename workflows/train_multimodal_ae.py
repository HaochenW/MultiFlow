#!/usr/bin/env python3
"""Train the 128-dimensional scDiffusion-X multimodal autoencoder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from multiflow_omics.h5mu import sha256_file, validate_h5mu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scdiffusion-x-root", required=True)
    parser.add_argument("--train-h5mu", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--encoder-config",
        default=str(Path(__file__).with_name("encoder_multimodal_128.yaml")),
    )
    parser.add_argument("--dataset-name", default="multiflow_data")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    repo = Path(args.scdiffusion_x_root).expanduser().resolve()
    train = Path(args.train_h5mu).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.encoder_config).expanduser().resolve()
    entry = repo / "script" / "training_autoencoder" / "train_encoder.py"
    official_config = (
        repo
        / "script"
        / "training_autoencoder"
        / "configs"
        / "encoder"
        / "encoder_multimodal.yaml"
    )
    for path in (train, config_path, entry, official_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = validate_h5mu(train)
    if not report["paired_raw_contract_ok"]:
        raise ValueError("multimodal AE requires raw-count RNA and binary ATAC")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rna_dims = list(config["encoder_kwargs"]["rna"]["dims"])
    atac_dims = list(config["encoder_kwargs"]["atac"]["dims"])
    if rna_dims[-1] != 128 or atac_dims[-1] != 128:
        raise ValueError("the paper workflow requires 128-dimensional RNA and ATAC latents")

    project = f"multiflow_{args.dataset_name}_ae128"
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_seeded_entry.py")),
        "--seed",
        str(args.seed),
        "--script",
        str(entry),
        "--",
        # The pinned scDiffusion-X checkout ships an openproblem dataset
        # schema. Every path and scale field is overridden below, so the
        # schema can also be used for another paired H5MU dataset.
        "dataset=openproblem",
        "~launcher",
        f"dataset.dataset_path={train}",
        "dataset.valid_path=null",
        "dataset.split_rates=[0.9,0.1]",
        "dataset.layer_key=X",
        "+dataset.encoder_type=learnt_autoencoder",
        "encoder=encoder_multimodal",
        "encoder.encoder_kwargs.rna.dims=" + json.dumps(rna_dims, separators=(",", ":")),
        "encoder.encoder_kwargs.atac.dims=" + json.dumps(atac_dims, separators=(",", ":")),
        "encoder.is_binarized=True",
        "checkpoints.every_n_epochs=20",
        "+checkpoints.save_top_k=-1",
        f"training_config.chekpoint_path={output}",
        f"logger.project={project}",
        "logger.offline=True",
        f"trainer.max_epochs={args.epochs}",
    ]
    env = os.environ.copy()
    source_root = str((repo / "scdiffusionX" / "src").resolve())
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    print("running: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=entry.parent, env=env, check=True)

    checkpoint_dir = output / project / "checkpoints"
    candidates = list(checkpoint_dir.glob("last*.ckpt")) or list(
        checkpoint_dir.glob("*.ckpt")
    )
    if not candidates:
        raise FileNotFoundError(f"no autoencoder checkpoint under {checkpoint_dir}")
    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    stable = output / "checkpoints" / "final.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, stable)
    state = {
        "schema_version": 1,
        "train_h5mu": str(train),
        "train_h5mu_sha256": sha256_file(train),
        "encoder_config": str(config_path),
        "encoder_config_sha256": sha256_file(config_path),
        "scdiffusion_x_root": str(repo),
        "rna_dims": rna_dims,
        "atac_dims": atac_dims,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "official_checkpoint": str(selected),
        "stable_checkpoint": str(stable),
        "stable_checkpoint_sha256": sha256_file(stable),
        "command": command,
    }
    (output / "multimodal_ae_training.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {stable}")


if __name__ == "__main__":
    main()
