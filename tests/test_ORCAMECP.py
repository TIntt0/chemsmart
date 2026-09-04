"""Tests for native ORCA SurfCrossOpt support."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from chemsmart.cli.orca.orca import orca
from chemsmart.cli.sub import sub
from chemsmart.io.orca.output import ORCAOutput
from chemsmart.jobs.orca.mecp import ORCAMECPJob
from chemsmart.jobs.orca.settings import ORCAMECPJobSettings
from chemsmart.jobs.orca.writer import ORCAInputWriter
from chemsmart.settings.orca import ORCAProjectSettings


def mecp_settings(**kwargs):
    values = {
        "charge": 1,
        "multiplicity_a": 6,
        "multiplicity_b": 4,
        "functional": "B3LYP",
        "basis": "TZVP",
    }
    values.update(kwargs)
    return ORCAMECPJobSettings(**values)


def test_route_and_validation():
    settings = mecp_settings()
    assert settings.route_string.startswith("! Opt SurfCrossOpt")
    assert settings.validate().multiplicity == 6
    with pytest.raises(ValueError, match="different"):
        mecp_settings(multiplicity_a=4, multiplicity_b=4).validate()


def test_numfreq_and_custom_route():
    settings = mecp_settings(
        mode="NumFreq", route_to_be_written="B3LYP TZVP TightSCF"
    )
    route = settings.route_string
    assert route.startswith("! Opt SurfCrossOpt SurfCrossNumFreq")
    assert "TightSCF" in route


def test_feo_official_example_input(
    tmpdir,
    orca_yaml_settings_orca_project_name,
    orca_jobrunner_no_scratch,
):
    project = ORCAProjectSettings.from_project(
        orca_yaml_settings_orca_project_name
    )
    settings = ORCAMECPJobSettings.from_settings(project.opt_settings())
    settings.charge = 1
    settings.multiplicity_a = 6
    settings.multiplicity_b = 4
    settings.maxiter = 200
    settings.validate()
    feo_xyz = Path("tests/data/ORCATests/inputs/xyz/feo_plus.xyz").resolve()
    job = ORCAMECPJob.from_filename(
        filename=str(feo_xyz),
        settings=settings,
        label="feo_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmpdir)
    content = Path(str(tmpdir), "feo_mecp.inp").read_text()
    assert "Opt SurfCrossOpt" in content
    assert "%mecp\n  Mult 4\nend" in content
    assert "%geom\n  MaxIter 200\nend" in content
    assert "* xyz 1 6" in content


def test_casscf_options_are_written(
    tmpdir, single_molecule_xyz_file, orca_jobrunner_no_scratch
):
    settings = mecp_settings(
        casscf_nel=6,
        casscf_norb=6,
        casscf_mult=[1, 3],
        casscf_nroots=[4, 2],
        casscf_bweight=[2, 1],
    )
    job = ORCAMECPJob.from_filename(
        filename=single_molecule_xyz_file,
        settings=settings,
        label="casscf_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmpdir)
    content = Path(str(tmpdir), "casscf_mecp.inp").read_text()
    assert "casscf_nel 6" in content
    assert "casscf_mult 1,3" in content
    assert "casscf_nroots 4,2" in content


def test_real_output_markers_are_parsed(tmp_path):
    output = tmp_path / "feo_mecp.out"
    output.write_text("""|  1> ! B3LYP TZVP Opt SurfCrossOpt SurfCrossNumFreq
|  2> %mecp Mult 4
FINAL SINGLE POINT ENERGY     -1340.123456789
FINAL SINGLE POINT ENERGY     -1340.123454391
Energy difference between both states      -0.000002398
THE OPTIMIZATION HAS CONVERGED
****ORCA TERMINATED NORMALLY****
""")
    parsed = ORCAOutput(str(output)).mecp_result
    assert parsed.converged
    assert parsed.normal_termination
    assert parsed.state_1_energy == pytest.approx(-1340.123456789)
    assert parsed.state_2_energy == pytest.approx(-1340.123454391)
    assert parsed.energy_gap == pytest.approx(-0.000002398)
    assert parsed.numfreq_requested


def test_real_feo_stationary_point_geometry():
    output = Path("tests/data/ORCATests/outputs/feo_plus_mecp.out").resolve()
    parsed = ORCAOutput(str(output)).mecp_result
    assert parsed.converged
    assert parsed.normal_termination
    assert parsed.final_structure.energy is None
    fe_o_distance = np.linalg.norm(
        parsed.final_structure.positions[0]
        - parsed.final_structure.positions[1]
    )
    assert fe_o_distance == pytest.approx(1.993648, abs=1e-6)


def test_numfreq_rejects_two_atom_system(orca_jobrunner_no_scratch):
    with pytest.raises(
        ValueError, match="SurfCrossNumFreq requires at least 3 atoms"
    ):
        ORCAMECPJob.from_filename(
            filename=str(
                Path("tests/data/ORCATests/inputs/xyz/feo_plus.xyz").resolve()
            ),
            settings=mecp_settings(mode="numfreq"),
            label="feo_numfreq",
            jobrunner=orca_jobrunner_no_scratch,
        )


def test_broken_symmetry_rejects_incompatible_pes2(
    orca_jobrunner_no_scratch,
):
    with pytest.raises(ValueError, match="generates PES2 multiplicity 1"):
        ORCAMECPJob.from_filename(
            filename=str(
                Path("tests/data/ORCATests/inputs/xyz/feo_plus.xyz").resolve()
            ),
            settings=mecp_settings(broken_sym=[1, 1]),
            label="invalid_feo_broken_symmetry",
            jobrunner=orca_jobrunner_no_scratch,
        )


def test_broken_symmetry_rejects_incompatible_electron_parity(
    orca_jobrunner_no_scratch,
):
    with pytest.raises(ValueError, match="incompatible.*33 electrons"):
        ORCAMECPJob.from_filename(
            filename=str(
                Path("tests/data/ORCATests/inputs/xyz/feo_plus.xyz").resolve()
            ),
            settings=mecp_settings(
                multiplicity_b=1,
                broken_sym=[1, 1],
            ),
            label="invalid_feo_broken_symmetry_parity",
            jobrunner=orca_jobrunner_no_scratch,
        )


def test_twisted_ethylene_broken_symmetry_input(
    tmp_path, orca_jobrunner_no_scratch
):
    job = ORCAMECPJob.from_filename(
        filename=str(
            Path(
                "tests/data/ORCATests/inputs/xyz/ethylene_twisted_mecp.xyz"
            ).resolve()
        ),
        settings=mecp_settings(
            charge=0,
            multiplicity_a=3,
            multiplicity_b=1,
            functional="B3LYP",
            basis="def2-SVP",
            scf_tol="TightSCF",
            broken_sym=[1, 1],
            maxiter=50,
        ),
        label="ethylene_twisted_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmp_path)
    input_text = (tmp_path / "ethylene_twisted_mecp.inp").read_text()
    assert "B3LYP def2-SVP TightSCF" in input_text
    assert "%mecp\n  Mult 1\n  brokenSym 1,1\nend" in input_text
    assert "* xyz 0 3" in input_text


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CHEMSMART_RUN_ORCA_INTEGRATION") != "1",
    reason="set CHEMSMART_RUN_ORCA_INTEGRATION=1 to run ORCA",
)
def test_real_orca_twisted_ethylene_mecp(tmp_path, orca_jobrunner_no_scratch):
    """Run an opt-in twisted-ethylene singlet/triplet MECP calculation.

    Set ``CHEMSMART_RUN_ORCA_INTEGRATION=1`` and, if needed,
    ``ORCA_EXE=/absolute/path/to/orca`` before invoking pytest.
    """
    executable = os.environ.get("ORCA_EXE") or shutil.which("orca")
    if executable is None:
        pytest.skip("ORCA_EXE is not set and orca is not on PATH")

    xyz = Path(
        "tests/data/ORCATests/inputs/xyz/ethylene_twisted_mecp.xyz"
    ).resolve()
    settings = mecp_settings(
        charge=0,
        multiplicity_a=3,
        multiplicity_b=1,
        functional="B3LYP",
        basis="def2-SVP",
        scf_tol="TightSCF",
        broken_sym=[1, 1],
        maxiter=50,
    )
    job = ORCAMECPJob.from_filename(
        filename=str(xyz),
        settings=settings,
        label="ethylene_twisted_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmp_path)
    input_path = tmp_path / "ethylene_twisted_mecp.inp"
    output_path = tmp_path / "ethylene_twisted_mecp.out"
    input_text = input_path.read_text()
    assert "%mecp\n  Mult 1\n  brokenSym 1,1\nend" in input_text
    assert "* xyz 0 3" in input_text

    with output_path.open("w") as output:
        completed = subprocess.run(
            [executable, input_path.name],
            cwd=tmp_path,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )

    output_text = output_path.read_text(errors="replace")
    assert completed.returncode == 0, output_text[-4000:]
    assert "multiplicity" not in output_text.lower() or (
        "impossible" not in output_text.lower()
    )
    assert "ORCA TERMINATED NORMALLY" in output_text
    assert "THE OPTIMIZATION HAS CONVERGED" in output_text


def test_orca_official_ch3o_ch2oh_numfreq_input(
    tmpdir, orca_jobrunner_no_scratch
):
    xyz = Path("tests/data/ORCATests/inputs/xyz/ch3o_ch2oh_mecp.xyz").resolve()
    settings = mecp_settings(
        charge=1,
        multiplicity_a=3,
        multiplicity_b=1,
        mode="numfreq",
    )
    job = ORCAMECPJob.from_filename(
        filename=str(xyz),
        settings=settings,
        label="ch3o_ch2oh_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmpdir)
    content = Path(str(tmpdir), "ch3o_ch2oh_mecp.inp").read_text()
    assert "Opt SurfCrossOpt SurfCrossNumFreq" in content
    assert "%mecp\n  Mult 1\nend" in content
    assert "* xyz 1 3" in content


def test_real_ch3o_ch2oh_numfreq_result():
    output = Path(
        "tests/data/ORCATests/outputs/ch3o_ch2oh_mecp_numfreq.out"
    ).resolve()
    result = ORCAOutput(str(output)).mecp_result
    assert result.numfreq_requested
    assert result.numfreq_completed
    assert result.is_minimum is True
    assert len(result.state_1_frequencies) == 15
    assert len(result.state_2_frequencies) == 15
    assert result.state_1_frequencies[7] == pytest.approx(775.23)
    assert result.state_2_frequencies[7] == pytest.approx(601.21)
    assert result.state_1_imaginary_frequencies == ()
    assert result.state_2_imaginary_frequencies == ()


def test_numfreq_imaginary_mode_is_not_minimum(tmp_path):
    source = Path(
        "tests/data/ORCATests/outputs/ch3o_ch2oh_mecp_numfreq.out"
    ).read_text()
    output = tmp_path / "mecp_imaginary.out"
    output.write_text(source.replace("601.21 cm**-1", "-42.00 cm**-1"))
    result = ORCAOutput(str(output)).mecp_result
    assert result.numfreq_completed
    assert result.state_2_imaginary_frequencies == (-42.0,)
    assert result.is_minimum is False


def test_mecp_solvent_constraint_and_moinp_are_written(
    tmpdir, tmp_path, orca_jobrunner_no_scratch
):
    xyz = Path("tests/data/ORCATests/inputs/xyz/ch3o_ch2oh_mecp.xyz").resolve()
    gbw = tmp_path / "pes2.gbw"
    gbw.write_bytes(b"test orbitals")
    settings = mecp_settings(
        charge=1,
        multiplicity_a=3,
        multiplicity_b=1,
        solvent_model="cpcm",
        solvent_id="water",
        moinp=str(gbw),
        maxiter=80,
        invert_constraints=True,
    )
    job = ORCAMECPJob.from_filename(
        filename=str(xyz),
        settings=settings,
        label="advanced_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    job.molecule.frozen_atoms = [-1, 0, 0, 0, 0]
    ORCAInputWriter(job=job).write(target_directory=tmpdir)
    content = Path(str(tmpdir), "advanced_mecp.inp").read_text()
    assert 'moinp "pes2.gbw"' in content
    assert Path(str(tmpdir), "pes2.gbw").read_bytes() == b"test orbitals"
    assert "CPCM(water)" in content
    assert content.count("%geom") == 1
    assert (
        "%geom\n  MaxIter 80\n  { C 0 C }\n  InvertConstraints True\nend"
        in content
    )


def test_cli_rejects_equal_multiplicities(
    single_molecule_xyz_file, run_orca_and_capture_settings
):
    result, _ = run_orca_and_capture_settings(
        "chemsmart.jobs.orca.mecp.ORCAMECPJob",
        [
            "-p",
            "test",
            "-f",
            single_molecule_xyz_file,
            "mecp",
            "--m1",
            "4",
            "--m2",
            "4",
            "--charge",
            "1",
        ],
    )
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "m1,m2,expected",
    [
        ("0", "4", "range x>=1"),
        ("-1", "4", "range x>=1"),
        ("6", "0", "range x>=1"),
        ("6", "-1", "range x>=1"),
    ],
)
def test_cli_rejects_nonpositive_multiplicities(
    single_molecule_xyz_file,
    run_orca_and_capture_settings,
    m1,
    m2,
    expected,
):
    result, _ = run_orca_and_capture_settings(
        "chemsmart.jobs.orca.mecp.ORCAMECPJob",
        [
            "-p",
            "test",
            "-f",
            single_molecule_xyz_file,
            "-c",
            "1",
            "mecp",
            "--m1",
            m1,
            "--m2",
            m2,
        ],
    )
    assert result.exit_code != 0
    assert expected in result.output


@pytest.mark.parametrize("args", [["--m1", "6"], ["--m2", "4"], []])
def test_cli_requires_both_multiplicities(
    single_molecule_xyz_file, run_orca_and_capture_settings, args
):
    result, _ = run_orca_and_capture_settings(
        "chemsmart.jobs.orca.mecp.ORCAMECPJob",
        [
            "-p",
            "test",
            "-f",
            single_molecule_xyz_file,
            "-c",
            "1",
            "mecp",
            *args,
        ],
    )
    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_orca_short_a_appends_label(
    single_molecule_xyz_file, orca_jobrunner_no_scratch
):
    with patch("chemsmart.jobs.orca.mecp.ORCAMECPJob") as job_class:
        job_class.return_value = object()
        result = CliRunner().invoke(
            orca,
            [
                "-p",
                "test",
                "-f",
                single_molecule_xyz_file,
                "-a",
                "testnumfreq",
                "-c",
                "1",
                "mecp",
                "--m1",
                "6",
                "--m2",
                "4",
                "--mode",
                "numfreq",
                "--freeze-atoms",
                "1",
            ],
            obj={"jobrunner": orca_jobrunner_no_scratch},
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    call = job_class.call_args.kwargs
    assert call["label"].endswith("_testnumfreq")
    assert call["settings"].aux_basis is None
    assert call["molecule"].frozen_atoms[0] == -1


def test_sub_preserves_mecp_arguments(orca_jobrunner_no_scratch):
    xyz = Path("tests/data/ORCATests/inputs/xyz/ch3o_ch2oh_mecp.xyz").resolve()
    server = orca_jobrunner_no_scratch.server
    with (
        patch(
            "chemsmart.cli.sub.Server.from_servername",
            return_value=server,
        ),
        patch.object(server, "submit") as submit,
        patch("chemsmart.jobs.orca.mecp.ORCAMECPJob") as job_class,
    ):
        job_class.return_value = MagicMock()
        result = CliRunner().invoke(
            sub,
            [
                "--server",
                "cuhk",
                "--test",
                "orca",
                "-p",
                "test",
                "-f",
                str(xyz),
                "-a",
                "testnumfreq",
                "-c",
                "1",
                "-x",
                "B3LYP",
                "-b",
                "TZVP",
                "mecp",
                "--m1",
                "3",
                "--m2",
                "1",
                "--mode",
                "numfreq",
                "--freeze-atoms",
                "1",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    submitted_args = submit.call_args.kwargs["cli_args"]
    for option, value in (
        ("--append-label", "testnumfreq"),
        ("--m1", "3"),
        ("--m2", "1"),
        ("--mode", "numfreq"),
        ("--freeze-atoms", "1"),
    ):
        index = submitted_args.index(option)
        assert submitted_args[index + 1] == value
    assert "--aux-basis" not in submitted_args


def test_cli_m1_m2_options(
    single_molecule_xyz_file,
    run_orca_and_capture_settings,
    orca_jobrunner_no_scratch,
):
    result, settings = run_orca_and_capture_settings(
        "chemsmart.jobs.orca.mecp.ORCAMECPJob",
        [
            "-p",
            "test",
            "-f",
            single_molecule_xyz_file,
            "-c",
            "1",
            "mecp",
            "--m1",
            "6",
            "--m2",
            "4",
        ],
        ctx_obj={"jobrunner": orca_jobrunner_no_scratch},
    )
    assert result.exit_code == 0, result.output
    assert settings.multiplicity == 6
    assert settings.multiplicity_a == 6
    assert settings.multiplicity_b == 4
