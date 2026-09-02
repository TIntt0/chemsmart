"""Tests for native ORCA SurfCrossOpt support."""

from pathlib import Path

import pytest

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
    settings.broken_sym = [1, 1]
    settings.validate()
    feo_xyz = Path(
        "tests/data/ORCATests/inputs/xyz/feo_plus.xyz"
    ).resolve()
    job = ORCAMECPJob.from_filename(
        filename=str(feo_xyz),
        settings=settings,
        label="feo_mecp",
        jobrunner=orca_jobrunner_no_scratch,
    )
    ORCAInputWriter(job=job).write(target_directory=tmpdir)
    content = Path(str(tmpdir), "feo_mecp.inp").read_text()
    assert "Opt SurfCrossOpt" in content
    assert "%mecp\n  Mult 4\n  brokenSym 1,1\nend" in content
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
    output.write_text(
        """|  1> ! B3LYP TZVP Opt SurfCrossOpt SurfCrossNumFreq
|  2> %mecp Mult 4
FINAL SINGLE POINT ENERGY     -1340.123456789
FINAL SINGLE POINT ENERGY     -1340.123454391
Energy difference between both states      -0.000002398
THE OPTIMIZATION HAS CONVERGED
****ORCA TERMINATED NORMALLY****
"""
    )
    parsed = ORCAOutput(str(output)).mecp_result
    assert parsed.converged
    assert parsed.normal_termination
    assert parsed.state_1_energy == pytest.approx(-1340.123456789)
    assert parsed.state_2_energy == pytest.approx(-1340.123454391)
    assert parsed.energy_gap == pytest.approx(-0.000002398)
    assert parsed.numfreq_requested


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
