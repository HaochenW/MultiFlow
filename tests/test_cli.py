from __future__ import annotations

import numpy as np
import pytest

from multiflow_omics import __version__
from multiflow_omics.cli import main


def test_cli_help_is_available_without_analysis_dependencies(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "train-latents" in output
    assert "sample-latents" in output
    assert "inspect-checkpoint" in output


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_tiny_train_inspect_and_sample_round_trip(tmp_path, capsys):
    source = tmp_path / "latents.npz"
    checkpoint = tmp_path / "model.pt"
    samples = tmp_path / "samples.npz"
    generator = np.random.default_rng(4)
    np.savez(
        source,
        rna=generator.normal(size=(4, 1)).astype(np.float32),
        atac=generator.normal(size=(4, 1)).astype(np.float32),
        labels=np.tile(np.arange(2), 2),
    )

    main(
        [
            "train-latents",
            "--input",
            str(source),
            "--output",
            str(checkpoint),
            "--model",
            "concat",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "2",
            "--num-layers",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".history.csv").is_file()

    main(["inspect-checkpoint", "--checkpoint", str(checkpoint)])
    assert '"model_type": "concat"' in capsys.readouterr().out

    main(
        [
            "sample-latents",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(samples),
            "--n",
            "2",
            "--label",
            "1",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )
    generated = np.load(samples)
    assert generated["rna"].shape == (2, 1)
    assert generated["atac"].shape == (2, 1)
    assert np.isfinite(generated["rna"]).all()
    assert np.isfinite(generated["atac"]).all()
