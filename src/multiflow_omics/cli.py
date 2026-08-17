"""User-facing command line interface for MultiFlow."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ._version import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .datasets import DATASETS, download_dataset
from .h5mu import (
    DEFAULT_ATAC_REPRESENTATION,
    DEFAULT_RNA_REPRESENTATION,
    read_paired_latents,
    sha256_file,
    validate_h5mu,
    write_generated_h5mu,
    write_toy_h5mu,
)
from .legacy import migrate_legacy_checkpoint
from .models import CellStateFlow, ConditionalConcatFlow, PerturbationFlow
from .normalization import LatentStandardizer
from .paper import (
    debias_perturbation_h5mu,
    decode_paper_h5mu,
    encode_paper_h5mu,
    prepare_perturbation_fold,
)
from .sampling import sample_paired_latents
from .training import TrainingConfig, fit, seed_everything


def _add_representation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rna-representation",
        default=DEFAULT_RNA_REPRESENTATION,
        help=f"RNA obsm key (default: {DEFAULT_RNA_REPRESENTATION})",
    )
    parser.add_argument(
        "--atac-representation",
        default=DEFAULT_ATAC_REPRESENTATION,
        help=f"ATAC obsm key (default: {DEFAULT_ATAC_REPRESENTATION})",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multiflow",
        description="Coupled flow matching for paired single-cell multiomics",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("data", help="download or validate paired H5MU data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    download = data_commands.add_parser("download", help="download a version-pinned dataset")
    download.add_argument("dataset", choices=sorted(DATASETS))
    download.add_argument("--output", required=True)
    download.add_argument("--accept-license", action="store_true")
    download.add_argument("--force", action="store_true")
    validate = data_commands.add_parser("validate", help="audit pairing and matrix scales")
    validate.add_argument("input")
    validate.add_argument("--condition-key", default="cell_type")
    _add_representation_arguments(validate)
    validate.add_argument("--json", dest="json_output", default=None)
    example = data_commands.add_parser("example", help="create a tiny paired H5MU file")
    example.add_argument("--output", default="toy_multiflow.h5mu")
    example.add_argument("--seed", type=int, default=7)

    train = subparsers.add_parser("train", help="train MultiFlow from paired H5MU latents")
    train.add_argument("--input", required=True, help="prepared paired .h5mu file")
    train.add_argument("--output", required=True, help="new run directory")
    _add_representation_arguments(train)
    train.add_argument("--condition-key", default="cell_type")
    train.add_argument("--unconditional", action="store_true")
    train.add_argument("--perturbation-key", default=None)
    train.add_argument("--context-matrix-key", default="multiflow_context_matrix")
    train.add_argument("--context-classes-key", default="multiflow_context_classes")
    train.add_argument(
        "--control-perturbation",
        default="control",
        help="perturbation name reserved as the zero vector (default: control)",
    )
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
    train.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help=(
            "default midpoint ODE steps saved with the run "
            "(default: 100 for generation, 50 for perturbation)"
        ),
    )
    train.add_argument(
        "--hash-input",
        action="store_true",
        help="record SHA256 of the input H5MU (can take time for large files)",
    )

    generate = subparsers.add_parser("generate", help="generate paired states as H5MU")
    generate.add_argument("--run", required=True, help="run directory created by train")
    generate.add_argument("--output", required=True, help="generated .h5mu file")
    generate.add_argument("--n", type=int, required=True)
    generate.add_argument("--cell-type", default=None)
    generate.add_argument("--perturbation", default=None)
    generate.add_argument("--steps", type=int, default=None)
    generate.add_argument("--batch-size", type=int, default=512)
    generate.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--keep-standardized", action="store_true")
    generate.add_argument(
        "--rng-mode",
        choices=["legacy_interleaved", "batch_invariant"],
        default="legacy_interleaved",
    )

    inspect = subparsers.add_parser("inspect", help="print checkpoint metadata")
    inspect.add_argument("--checkpoint", required=True)

    migrate = subparsers.add_parser(
        "migrate",
        help="convert a trusted historical checkpoint to the portable format",
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
        help="optional legacy NPZ containing rna_mean/rna_std/atac_mean/atac_std",
    )
    migrate.add_argument(
        "--label-classes",
        nargs="+",
        default=None,
        help="ordered biological names for legacy cell-type/context indices",
    )
    migrate.add_argument(
        "--perturbation-classes",
        nargs="+",
        default=None,
        help="ordered perturbation names; index 0 must be the control",
    )
    migrate.add_argument("--trust-source", action="store_true")

    paper = subparsers.add_parser(
        "paper",
        help="run the task-specific raw-profile encoder and decoder workflow",
    )
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)
    encode = paper_commands.add_parser(
        "encode",
        help="encode raw RNA/ATAC profiles with the paper encoders",
    )
    encode.add_argument("--task", choices=["generation", "perturbation"], required=True)
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--scdiffusion-x-root", required=True)
    encode.add_argument("--multimodal-ae-checkpoint", required=True)
    encode.add_argument("--encoder-config", required=True)
    encode.add_argument("--rna-vae-checkpoint", default=None)
    encode.add_argument("--condition-key", default="cell_type")
    encode.add_argument("--batch-size", type=int, default=2048)
    encode.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    fold = paper_commands.add_parser(
        "prepare-perturbation-fold",
        help="remove held-out non-control cells and build the notebook context matrix",
    )
    fold.add_argument("--input", required=True)
    fold.add_argument("--output", required=True)
    fold.add_argument("--held-out-cell-type", required=True)
    fold.add_argument("--condition-key", default="cell_type")
    fold.add_argument("--perturbation-key", default="perturbation")
    fold.add_argument("--control-perturbation", default="control")

    debias = paper_commands.add_parser(
        "debias-perturbation",
        help="apply the paper training-only perturbation mean-shift correction",
    )
    debias.add_argument("--input", required=True, help="generated latent H5MU")
    debias.add_argument("--training-fold", required=True)
    debias.add_argument("--output", required=True)
    debias.add_argument("--held-out-cell-type", required=True)
    debias.add_argument("--perturbation", required=True)
    debias.add_argument("--condition-key", default="cell_type")
    debias.add_argument("--perturbation-key", default="perturbation")
    debias.add_argument("--control-perturbation", default="control")
    debias.add_argument("--alpha", type=float, default=1.0)
    debias.add_argument("--min-cells", type=int, default=5)

    decode = paper_commands.add_parser(
        "decode",
        help="decode generated latents with the matching paper decoders",
    )
    decode.add_argument("--task", choices=["generation", "perturbation"], required=True)
    decode.add_argument("--input", required=True, help="generated latent H5MU")
    decode.add_argument("--reference", required=True, help="encoded training H5MU")
    decode.add_argument("--output", required=True)
    decode.add_argument("--scdiffusion-x-root", required=True)
    decode.add_argument("--multimodal-ae-checkpoint", required=True)
    decode.add_argument("--encoder-config", required=True)
    decode.add_argument("--rna-vae-checkpoint", default=None)
    decode.add_argument("--condition-key", default="cell_type")
    decode.add_argument("--batch-size", type=int, default=2048)
    decode.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    decode.add_argument("--seed", type=int, default=0)
    return parser


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_directory(path: str) -> Path:
    run = Path(path).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")
    run.mkdir(parents=True, exist_ok=True)
    return run


def _check_run_target(path: str) -> None:
    run = Path(path).expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")


def _default_sampling_steps(model_type: str) -> int:
    """Return the notebook-aligned midpoint solver budget."""
    return 50 if model_type == "perturbation" else 100


def _train(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    if input_path.suffix.lower() != ".h5mu":
        raise ValueError("MultiFlow training input must be a paired .h5mu file")
    _check_run_target(args.output)
    condition_key = None if args.unconditional else args.condition_key
    data = read_paired_latents(
        input_path,
        rna_representation=args.rna_representation,
        atac_representation=args.atac_representation,
        condition_key=condition_key,
        perturbation_key=args.perturbation_key,
        context_matrix_key=args.context_matrix_key,
        context_classes_key=args.context_classes_key,
        control_perturbation=args.control_perturbation,
    )
    if args.model == "perturbation" and args.perturbation_key is None:
        raise ValueError("--model perturbation requires --perturbation-key")
    if args.model != "perturbation" and args.perturbation_key is not None:
        raise ValueError("--perturbation-key is accepted only by --model perturbation")
    seed_everything(args.seed)
    num_classes = None if data.labels is None else len(data.label_classes)
    if args.model == "cell-state":
        model = CellStateFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            num_classes=num_classes,
            cross_attention_dim=args.cross_attention_dim,
        )
    elif args.model == "perturbation":
        if data.context_matrix is None:
            raise ValueError("perturbation training requires a context matrix")
        model = PerturbationFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            torch.as_tensor(data.context_matrix),
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            cross_attention_dim=args.cross_attention_dim,
            num_perturbations=len(data.perturbation_classes),
        )
    else:
        model = ConditionalConcatFlow(
            data.rna.shape[1],
            data.atac.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
        )

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        standardize=not args.no_standardize,
        reseed=False,
    )
    result = fit(
        model,
        data.rna,
        data.atac,
        labels=data.labels,
        perturbations=data.perturbations,
        config=config,
    )
    run = _run_directory(args.output)
    input_metadata: dict[str, object] = {
        "name": input_path.name,
        "size_bytes": input_path.stat().st_size,
    }
    if args.hash_input:
        input_metadata["sha256"] = sha256_file(input_path)
    default_sampling_steps = (
        int(args.sampling_steps)
        if args.sampling_steps is not None
        else _default_sampling_steps(args.model)
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "input": input_metadata,
        "h5mu_contract": {
            "rna_representation": args.rna_representation,
            "atac_representation": args.atac_representation,
            "condition_key": condition_key,
            "perturbation_key": args.perturbation_key,
            "control_perturbation": (
                args.control_perturbation if args.perturbation_key is not None else None
            ),
            "paired_cells": int(data.rna.shape[0]),
        },
        "label_classes": data.label_classes,
        "perturbation_classes": data.perturbation_classes,
        "training": asdict(config),
        "sampling": {"default_steps": default_sampling_steps},
    }
    checkpoint = save_checkpoint(
        run / "model.pt",
        result.model,
        standardizer=result.standardizer,
        optimizer=result.optimizer,
        epoch=args.epochs,
        history=result.history,
        metadata=metadata,
    )
    history_path = run / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.history[0].keys())
        writer.writeheader()
        writer.writerows(result.history)
    manifest = {
        "package_version": __version__,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": result.model.get_config(),
        **metadata,
    }
    _write_json(run / "run.json", manifest)
    print(f"saved run: {run}")
    print(f"  checkpoint: {checkpoint.name}")
    print(f"  history: {history_path.name}")
    print("  manifest: run.json")


def _resolve_named_condition(
    value: str | None,
    classes: list[str],
    *,
    name: str,
    required: bool,
) -> tuple[int | None, str | None]:
    if not classes:
        if value is not None:
            raise ValueError(f"--{name} was provided to an unconditional run")
        return None, None
    if value is None:
        if required:
            raise ValueError(f"--{name} is required; choose from {classes}")
        return 0, classes[0]
    try:
        index = classes.index(value)
    except ValueError as exc:
        raise ValueError(f"unknown {name} {value!r}; choose from {classes}") from exc
    return index, value


def _generate(args: argparse.Namespace) -> None:
    run = Path(args.run).expanduser().resolve()
    checkpoint = run / "model.pt" if run.is_dir() else run
    model, standardizer, payload = load_checkpoint(checkpoint, map_location=args.device)
    metadata = payload.get("metadata", {})
    label_classes = list(metadata.get("label_classes", []))
    perturbation_classes = list(metadata.get("perturbation_classes", []))
    model_config = model.get_config()
    expected_labels = (
        model_config.get("num_contexts")
        if model_config.get("model_type") == "perturbation"
        else model_config.get("num_classes")
    )
    expected_perturbations = (
        model_config.get("num_perturbations")
        if model_config.get("model_type") == "perturbation"
        else None
    )
    for mapping_name, classes, expected in (
        ("label_classes", label_classes, expected_labels),
        ("perturbation_classes", perturbation_classes, expected_perturbations),
    ):
        if expected is None or (mapping_name == "perturbation_classes" and int(expected) == 1):
            continue
        if len(classes) != int(expected):
            raise ValueError(
                f"checkpoint requires {int(expected)} {mapping_name}, but contains "
                f"{len(classes)}; remigrate the checkpoint with --{mapping_name.replace('_', '-')}"
            )
    label_code, label_name = _resolve_named_condition(
        args.cell_type,
        label_classes,
        name="cell-type",
        required=bool(label_classes),
    )
    perturbation_code, perturbation_name = _resolve_named_condition(
        args.perturbation,
        perturbation_classes,
        name="perturbation",
        required=bool(perturbation_classes),
    )
    fallback_steps = _default_sampling_steps(str(model_config.get("model_type")))
    default_steps = int(metadata.get("sampling", {}).get("default_steps", fallback_steps))
    steps = default_steps if args.steps is None else int(args.steps)
    standardizer_present = standardizer is not None
    standardization_inverted = standardizer_present and not args.keep_standardized
    if not standardizer_present and "legacy_migration" in metadata:
        warnings.warn(
            "legacy checkpoint has no latent standardizer; generated values remain "
            "in the model's recorded output space",
            RuntimeWarning,
            stacklevel=2,
        )
    rna, atac = sample_paired_latents(
        model,
        args.n,
        label=label_code,
        perturbation=perturbation_code,
        steps=steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        standardizer=standardizer if standardization_inverted else None,
        rng_mode=args.rng_mode,
    )
    output_metadata = {
        "schema_version": 1,
        "package_version": __version__,
        "output_space": "latent",
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": model_config,
        "seed": int(args.seed),
        "ode_steps": steps,
        "batch_size": int(args.batch_size),
        "rng_mode": args.rng_mode,
        "standardizer_present": standardizer_present,
        "standardization_inverted": standardization_inverted,
        "label_classes": label_classes,
        "perturbation_classes": perturbation_classes,
    }
    destination = write_generated_h5mu(
        args.output,
        rna.numpy(),
        atac.numpy(),
        cell_type=label_name,
        cell_type_code=label_code,
        perturbation=perturbation_name,
        perturbation_code=perturbation_code,
        metadata=output_metadata,
    )
    print(f"saved paired generated H5MU: {destination}")


def _validate_data(args: argparse.Namespace) -> None:
    report = validate_h5mu(
        args.input,
        condition_key=args.condition_key,
        rna_representation=args.rna_representation,
        atac_representation=args.atac_representation,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        destination = Path(args.json_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    if not report["paired_raw_contract_ok"]:
        raise ValueError("H5MU raw matrix scale contract failed")
    if not report["latent_ready"]:
        print(
            "note: raw paired data are valid, but encoder latents are absent; "
            "run `multiflow paper encode` before flow training"
        )


def _download_data(args: argparse.Namespace) -> None:
    dataset = DATASETS[args.dataset]
    print(f"source DOI: https://doi.org/{dataset.doi}")
    print(f"license: {dataset.license}")
    print(f"download size: {dataset.size_bytes / 1e9:.2f} GB")
    destination = download_dataset(
        args.dataset,
        args.output,
        accept_license=args.accept_license,
        force=args.force,
    )
    print(f"saved verified dataset: {destination}")


def _example_data(args: argparse.Namespace) -> None:
    destination = write_toy_h5mu(args.output, seed=args.seed)
    print(f"saved paired H5MU example: {destination}")


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
    migration_metadata: dict[str, object] = {}
    if args.label_classes is not None:
        migration_metadata["label_classes"] = list(args.label_classes)
    if args.perturbation_classes is not None:
        migration_metadata["perturbation_classes"] = list(args.perturbation_classes)
    output = migrate_legacy_checkpoint(
        args.source,
        args.output,
        model_type=None if args.model_type is None else args.model_type.replace("-", "_"),
        standardizer=standardizer,
        trust_source=args.trust_source,
        metadata=migration_metadata,
    )
    print(f"saved migrated checkpoint: {output}")
    if standardizer is None:
        _, recovered, _ = load_checkpoint(output)
        if recovered is None:
            print(
                "warning: no latent standardizer was available; supply the original "
                "training-only statistics before decoding generated latents"
            )


def _paper(args: argparse.Namespace) -> None:
    if args.paper_command == "encode":
        output = encode_paper_h5mu(
            input_path=args.input,
            output_path=args.output,
            task=args.task,
            scdiffusionx_root=args.scdiffusion_x_root,
            multimodal_ae_checkpoint=args.multimodal_ae_checkpoint,
            encoder_config=args.encoder_config,
            rna_vae_checkpoint=args.rna_vae_checkpoint,
            condition_key=args.condition_key,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"saved encoded paper input: {output}")
    elif args.paper_command == "prepare-perturbation-fold":
        output = prepare_perturbation_fold(
            input_path=args.input,
            output_path=args.output,
            held_out_cell_type=args.held_out_cell_type,
            condition_key=args.condition_key,
            perturbation_key=args.perturbation_key,
            control_perturbation=args.control_perturbation,
        )
        print(f"saved perturbation training fold: {output}")
    elif args.paper_command == "debias-perturbation":
        output = debias_perturbation_h5mu(
            generated_path=args.input,
            training_fold_path=args.training_fold,
            output_path=args.output,
            held_out_cell_type=args.held_out_cell_type,
            perturbation=args.perturbation,
            condition_key=args.condition_key,
            perturbation_key=args.perturbation_key,
            control_perturbation=args.control_perturbation,
            alpha=args.alpha,
            min_cells=args.min_cells,
        )
        print(f"saved debiased perturbation latents: {output}")
    elif args.paper_command == "decode":
        output = decode_paper_h5mu(
            generated_path=args.input,
            reference_path=args.reference,
            output_path=args.output,
            task=args.task,
            scdiffusionx_root=args.scdiffusion_x_root,
            multimodal_ae_checkpoint=args.multimodal_ae_checkpoint,
            encoder_config=args.encoder_config,
            rna_vae_checkpoint=args.rna_vae_checkpoint,
            condition_key=args.condition_key,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"saved decoded RNA/ATAC profiles: {output}")
    else:  # pragma: no cover - argparse restricts the accepted subcommands.
        raise ValueError(f"unsupported paper command: {args.paper_command}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "data" and args.data_command == "download":
        _download_data(args)
    elif args.command == "data" and args.data_command == "validate":
        _validate_data(args)
    elif args.command == "data":
        _example_data(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "generate":
        _generate(args)
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "paper":
        _paper(args)
    else:
        _migrate(args)


if __name__ == "__main__":
    main()
