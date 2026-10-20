"""Tests for the MLIP calculator adapter and CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase.calculators.emt import EMT
from click.testing import CliRunner

from chemsmart.calculators.mlip import MLIPCalculator
from chemsmart.cli.mlip import mlip


@pytest.fixture
def h2_xyz(tmp_path):
    structure = tmp_path / "h2.xyz"
    structure.write_text(
        "2\nH2\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8"
    )
    return structure


def test_builds_pretrained_mace_calculator():
    calculator = MagicMock()
    mace_mp = MagicMock(return_value=calculator)
    with patch.dict(
        "sys.modules",
        {
            "mace": MagicMock(),
            "mace.calculators": MagicMock(
                mace_mp=mace_mp,
                MACECalculator=MagicMock(),
            ),
        },
    ):
        adapter = MLIPCalculator(pretrained="medium", device="cuda")
        assert adapter.build() is calculator

    mace_mp.assert_called_once_with(
        model="medium",
        device="cuda",
        default_dtype="float64",
        dispersion=False,
    )


def test_local_checkpoint_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MLIPCalculator(checkpoint=tmp_path / "missing.model")


def test_model_sources_are_mutually_exclusive(tmp_path):
    checkpoint = tmp_path / "model.model"
    checkpoint.touch()
    with pytest.raises(ValueError, match="either"):
        MLIPCalculator(pretrained="small", checkpoint=checkpoint)


def test_sp_writes_energy_and_forces(h2_xyz, tmp_path):
    with patch.object(MLIPCalculator, "build", return_value=EMT()):
        result = CliRunner().invoke(
            mlip,
            ["-f", str(h2_xyz), "-o", str(tmp_path), "sp"],
        )

    assert result.exit_code == 0, result.output
    text = (tmp_path / "mlip_sp.json").read_text(encoding="utf-8")
    assert "energy_eV" in text
    assert "forces_eV_per_Angstrom" in text


def test_opt_writes_optimized_structure(h2_xyz, tmp_path):
    with patch.object(MLIPCalculator, "build", return_value=EMT()):
        result = CliRunner().invoke(
            mlip,
            [
                "-f",
                str(h2_xyz),
                "-o",
                str(tmp_path),
                "opt",
                "--steps",
                "2",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "mlip_opt.xyz").is_file()
    assert (tmp_path / "mlip_opt.json").is_file()


def test_md_writes_trajectory_and_energies(h2_xyz, tmp_path):
    with patch.object(MLIPCalculator, "build", return_value=EMT()):
        result = CliRunner().invoke(
            mlip,
            [
                "-f",
                str(h2_xyz),
                "-o",
                str(tmp_path),
                "md",
                "--steps",
                "2",
                "--sample-interval",
                "1",
                "--seed",
                "7",
            ],
        )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "mlip_md.traj").is_file()
    energies = tmp_path / "mlip_md.csv"
    assert energies.is_file()
    rows = energies.read_text(encoding="utf-8").splitlines()
    assert len([row for row in rows if row.startswith("0,")]) == 1
    assert (tmp_path / "mlip_md_final.xyz").is_file()


def test_cli_rejects_two_model_sources(h2_xyz, tmp_path):
    checkpoint = Path(tmp_path / "model.model")
    checkpoint.touch()
    result = CliRunner().invoke(
        mlip,
        [
            "-f",
            str(h2_xyz),
            "--pretrained",
            "small",
            "--checkpoint",
            str(checkpoint),
            "sp",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
