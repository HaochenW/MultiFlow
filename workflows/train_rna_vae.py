#!/usr/bin/env python3
"""Fine-tune the paper RNA VAE on raw-count RNA from a training H5MU."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mudata as mu
import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multiflow_omics.h5mu import sha256_file
from multiflow_omics.paper_encoders import (
    RNAExpressionVAE,
    load_scimilarity_initialization,
)


class Rows(Dataset[torch.Tensor]):
    def __init__(self, matrix: object) -> None:
        self.matrix = matrix

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.matrix[index]
        if sparse.issparse(row):
            row = row.toarray()
        return torch.as_tensor(np.asarray(row, dtype=np.float32).reshape(-1))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5mu", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scimilarity-model-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source = Path(args.train_h5mu).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    mdata = mu.read_h5mu(source)
    adata = mdata["rna"].copy()
    original_names = list(map(str, adata.var_names))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if original_names != list(map(str, adata.var_names)):
        raise RuntimeError("RNA normalization changed feature order")

    model = RNAExpressionVAE(n_genes=adata.n_vars, latent_dim=128)
    load_scimilarity_initialization(
        model,
        args.scimilarity_model_dir,
        device=args.device,
    )
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.MSELoss(reduction="mean")
    loader = DataLoader(
        Rows(adata.X),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    iterator = iter(loader)
    last_checkpoint: Path | None = None
    for step in range(args.max_steps):
        try:
            genes = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            genes = next(iterator)
        genes = genes.to(args.device)
        reconstruction = model(genes)
        loss = criterion(reconstruction, genes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1_000 == 0:
            print(f"step={step} mse={loss.item():.8f}", flush=True)
        save = step % args.checkpoint_every == 0 or step == args.max_steps - 1
        if save:
            last_checkpoint = output / f"rna_vae_step_{step:06d}.pt"
            torch.save(model.state_dict(), last_checkpoint)
            print(f"saved {last_checkpoint}", flush=True)

    assert last_checkpoint is not None
    metadata = {
        "schema_version": 1,
        "train_h5mu": str(source),
        "train_h5mu_sha256": sha256_file(source),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "rna_input_scale": "normalize_total(1e4)+log1p",
        "latent_dim": 128,
        "hidden_dims": [1024, 1024, 1024],
        "optimizer": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "checkpoint": str(last_checkpoint),
        "checkpoint_sha256": sha256_file(last_checkpoint),
    }
    (output / "rna_vae_training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

