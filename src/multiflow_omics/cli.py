"""Command-line interface for latent-level MultiFlow workflows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ._version import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .legacy import migrate_legacy_checkpoint
from .models import CellStateFlow, ConditionalConcatFlow, PerturbationFlow
from .normalization import LatentStandardizer
from .sampling import sample_paired_latents
from .training import TrainingConfig, fit, seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multiflow-omics", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-latents", help="fit MultiFlow to a paired NPZ file")
    train.add_argument("--input", required=True, help="NPZ with rna, atac, and condition arrays")
    train.add_argument("--output", required=True, help="output checkpoint")
    train.add_argument(
        "--model",
        choices=["cell-state", "perturbation", "concat"],
        default="cell-state",
    )
    train.add_argument("--epochs", type=int, default=600)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--num-blocks", type=int, default=2)
    train.add_argument("--num-layers", type=int, default=8)
    train.add_argument("--cross-attention-dim", type=int, default=64)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--no-standardize", action="store_true")

    sample = subparsers.add_parser("sample-latents", help="sample paired latents")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--n", type=int, required=True)
    sample.add_argument("--label", type=int, default=None)
    sample.add_argument("--perturbation", type=int, default=None)
    sample.add_argument("--steps", type=int, default=100)
    sample.add_argument("--batch-size", type=int, default=512)
    sample.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--keep-standardized", action="store_true")
    sample.add_argument(
        "--rng-mode",
        choices=["legacy_interleaved", "batch_invariant"],
        default="legacy_interleaved",
    )

    inspect = subparsers.add_parser("inspect-checkpoint", help="print checkpoint metadata")
    inspect.add_argument("--checkpoint", required=True)

    migrate = subparsers.add_parser(
        "migrate-checkpoint",
        help="convert a trusted historical checkpoint to the safe format",
    )
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument(
        "--model-type",
        choices=["cell-state", "perturbation", "concat"],
        default=None,
    )
    migrate.add_argument(
        "--standardizer-npz",
        default=None,
        help="optional NPZ containing rna_mean, rna_std, atac_mean, atac_std",
    )
    migrate.add_argument(
        "--trust-source",
        action="store_true",
        help="allow unsafe pickle loading only after independently verifying the file",
    )
    return parser


def _train(args: argparse.Namespace) -> None:
    # Model parameters are initialized before fit() resets the training RNG,
    # so seed here as well to make the complete CLI run reproducible.
    seed_everything(args.seed)
    source = np.load(args.input)
    if not {"rna", "atac"}.issubset(source.files):
        raise ValueError("input NPZ must contain rna and atac arrays")
    rna, atac = source["rna"], source["atac"]
    labels = source["labels"] if "labels" in source.files else None
    perturbations = source["perturbations"] if "perturbations" in source.files else None
    num_classes = None if labels is None else int(np.max(labels)) + 1
    if args.model == "cell-state":
        model = CellStateFlow(
            rna.shape[1],
            atac.shape[1],
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            num_classes=num_classes,
            cross_attention_dim=args.cross_attention_dim,
        )
    elif args.model == "perturbation":
        if labels is None or perturbations is None or "context_matrix" not in source.files:
            raise ValueError(
                "perturbation training requires labels, perturbations, and context_matrix"
            )
        model = PerturbationFlow(
            rna.shape[1],
            atac.shape[1],
            torch.as_tensor(source["context_matrix"]),
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            cross_attention_dim=args.cross_attention_dim,
            num_perturbations=int(np.max(perturbations)) + 1,
        )
    else:
        model = ConditionalConcatFlow(
            rna.shape[1],
            atac.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
        )
    result = fit(
        model,
        rna,
        atac,
        labels=labels,
        perturbations=perturbations,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            standardize=not args.no_standardize,
        ),
    )
    output = save_checkpoint(
        args.output,
        result.model,
        standardizer=result.standardizer,
        optimizer=result.optimizer,
        epoch=args.epochs,
        history=result.history,
        metadata={"input_name": Path(args.input).name},
    )
    history_path = output.with_suffix(".history.csv")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.history[0].keys())
        writer.writeheader()
        writer.writerows(result.history)
    print(f"saved checkpoint: {output}")
    print(f"saved history: {history_path}")


def _sample(args: argparse.Namespace) -> None:
    model, standardizer, _ = load_checkpoint(args.checkpoint, map_location=args.device)
    rna, atac = sample_paired_latents(
        model,
        args.n,
        label=args.label,
        perturbation=args.perturbation,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        standardizer=None if args.keep_standardized else standardizer,
        rng_mode=args.rng_mode,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, rna=rna.numpy(), atac=atac.numpy())
    print(f"saved paired latents: {destination}")


def _inspect(args: argparse.Namespace) -> None:
    model, standardizer, payload = load_checkpoint(args.checkpoint)
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "format_version": payload["format_version"],
        "package_version": payload.get("package_version"),
        "model_config": model.get_config(),
        "epoch": payload.get("epoch"),
        "has_standardizer": standardizer is not None,
        "metadata": payload.get("metadata", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _migrate(args: argparse.Namespace) -> None:
    standardizer = None
    if args.standardizer_npz is not None:
        with np.load(args.standardizer_npz) as source:
            required = {"rna_mean", "rna_std", "atac_mean", "atac_std"}
            missing = sorted(required.difference(source.files))
            if missing:
                raise ValueError(f"standardizer NPZ is missing: {missing}")
            standardizer = LatentStandardizer.from_state_dict(
                {name: torch.as_tensor(source[name]) for name in required}
            )
    output = migrate_legacy_checkpoint(
        args.source,
        args.output,
        model_type=None if args.model_type is None else args.model_type.replace("-", "_"),
        standardizer=standardizer,
        trust_source=args.trust_source,
    )
    print(f"saved migrated checkpoint: {output}")
    if standardizer is None:
        _, recovered, _ = load_checkpoint(output)
        if recovered is None:
            print(
                "warning: no latent standardizer was available; provide the original "
                "training-only statistics before decoding generated latents"
            )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train-latents":
        _train(args)
    elif args.command == "sample-latents":
        _sample(args)
    elif args.command == "inspect-checkpoint":
        _inspect(args)
    else:
        _migrate(args)


if __name__ == "__main__":
    main()
