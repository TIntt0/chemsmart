from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from chemsmart.cli.main import entry_point
from chemsmart.cli.run import run
from chemsmart.cli.sub import sub
from chemsmart.settings.server import Server


def test_gaussian_directory_modred_submits_single_batch(tmp_path, monkeypatch):
    directory = tmp_path / "structures"
    directory.mkdir()
    for name, distance in [("b", 0.8), ("a", 0.7)]:
        (directory / f"{name}.xyz").write_text(
            f"2\n{name}\nH 0 0 0\nH 0 0 {distance}\n"
        )
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=64, MEM_GB=192, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.from_servername", return_value=server),
        patch.object(server, "submit_batch_job") as submit,
    ):
        result = CliRunner().invoke(
            entry_point,
            [
                "sub",
                "-s",
                "cuhk",
                "-N",
                "2",
                "-n",
                "64",
                "-m",
                "180",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory) + "/",
                "-c",
                "0",
                "-m",
                "1",
                "modred",
                "-c",
                "[[1,2]]",
            ],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    submit.assert_called_once()
    batch = submit.call_args.kwargs
    assert batch["num_nodes"] == 2
    assert len(batch["jobs"]) == 2
    assert all(
        job.jobrunner.num_cores == 64 and job.jobrunner.mem_gb == 180
        for job in batch["jobs"]
    )
    assert [
        Path(args[args.index("--filename") + 1]).name
        for args in batch["cli_args"]
    ] == ["a.xyz", "b.xyz"]
    assert len({job.label for job in batch["jobs"]}) == 2
    for job, args in zip(batch["jobs"], batch["cli_args"]):
        assert args[args.index("--batch-label") + 1] == job.label
        assert "[[1,2]]" in args
        captured = []

        def capture(actual, runner):
            captured.append(actual)
            actual.is_complete = lambda: True

        with (
            patch("chemsmart.cli.run.JobRunner") as runner,
            patch("chemsmart.cli.run._run_single_job", side_effect=capture),
        ):
            runner.return_value = job.jobrunner
            replay = CliRunner().invoke(run, args)
        assert replay.exit_code == 0, (replay.output, replay.exception)
        assert captured[0].label == job.label
        assert captured[0].settings.charge == 0
        assert captured[0].settings.multiplicity == 1
        assert captured[0].settings.modred == job.settings.modred


def test_directory_input_rejects_unsupported_files_before_submission(
    tmp_path, monkeypatch
):
    directory = tmp_path / "structures"
    directory.mkdir()
    (directory / "molecule.xyz").write_text(
        "2\nhydrogen\nH 0 0 0\nH 0 0 0.7\n"
    )
    (directory / "unexpected.txt").write_text("not a structure")
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=64, MEM_GB=192, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.current", return_value=server),
        patch.object(server, "submit_batch_job") as submit,
    ):
        result = CliRunner().invoke(
            sub,
            [
                "-N",
                "2",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory),
                "-c",
                "0",
                "-m",
                "1",
                "modred",
                "-c",
                "[[1,2]]",
            ],
        )
    assert result.exit_code != 0
    assert "Unsupported file in input directory" in result.output
    submit.assert_not_called()


def test_empty_directory_is_not_submitted(tmp_path, monkeypatch):
    directory = tmp_path / "empty"
    directory.mkdir()
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=64, MEM_GB=192, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.current", return_value=server),
        patch.object(server, "submit_batch_job") as submit,
    ):
        result = CliRunner().invoke(
            sub,
            [
                "-N",
                "2",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory),
                "-c",
                "0",
                "-m",
                "1",
                "modred",
                "-c",
                "[[1,2]]",
            ],
        )
    assert result.exit_code != 0
    assert "contains no files" in result.output
    submit.assert_not_called()


def test_full_cli_directory_requires_num_nodes(tmp_path, monkeypatch):
    directory = tmp_path / "structures"
    directory.mkdir()
    (directory / "molecule.xyz").write_text("1\nhelium\nHe 0 0 0\n")
    server = Server("test", SCHEDULER="SLURM", NUM_CORES=8, MEM_GB=24)
    monkeypatch.chdir(tmp_path)
    with patch("chemsmart.cli.sub.Server.current", return_value=server):
        result = CliRunner().invoke(
            entry_point,
            [
                "sub",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory),
                "sp",
            ],
        )
    assert result.exit_code != 0
    assert "requires sub -N" in result.output


def test_full_cli_xtb_npz_without_index_prepares_all_structures(
    tmp_path, monkeypatch
):
    source = Path(__file__).parent / "data/IOTests/revQM9_subset.npz"
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=8, MEM_GB=24, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.current", return_value=server),
        patch.object(server, "submit_batch_job") as submit,
    ):
        result = CliRunner().invoke(
            entry_point,
            [
                "sub",
                "-N",
                "2",
                "--test",
                "xtb",
                "-p",
                "test",
                "-f",
                str(source),
                "-c",
                "0",
                "-m",
                "1",
                "sp",
            ],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(submit.call_args.kwargs["jobs"]) == 5


def test_directory_writes_one_slurm_script_and_manifest(tmp_path, monkeypatch):
    import json

    from chemsmart.settings.submitters import SLURMSubmitter

    directory = tmp_path / "structures"
    directory.mkdir()
    for name in ("first", "second"):
        (directory / f"{name}.xyz").write_text(
            "2\nhydrogen\nH 0 0 0\nH 0 0 0.7\n"
        )
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=64, MEM_GB=192, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.current", return_value=server),
        patch(
            "chemsmart.utils.cluster.ClusterHelper.get_running_jobs",
            return_value=([], []),
        ),
        patch.object(SLURMSubmitter, "_write_program_specifics"),
    ):
        result = CliRunner().invoke(
            sub,
            [
                "-N",
                "2",
                "-n",
                "64",
                "--mem-gb",
                "180",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory),
                "-c",
                "0",
                "-m",
                "1",
                "modred",
                "-c",
                "[[1,2]]",
            ],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    scripts = list(tmp_path.glob("*.sh"))
    manifests = list(tmp_path.glob("*.json"))
    assert len(scripts) == 1
    assert len(manifests) == 1
    assert "#SBATCH --nodes=2" in scripts[0].read_text()
    manifest = json.loads(manifests[0].read_text())
    assert manifest["task_cores"] == 64
    assert manifest["task_memory_gb"] == 180
    assert len(manifest["tasks"]) == 2


def test_directory_replay_preserves_gaussian_input_file_settings(
    tmp_path, monkeypatch
):
    import shutil

    source = (
        Path(__file__).parent
        / "data/GaussianTests/inputs/model_modred_input.com"
    )
    directory = tmp_path / "structures"
    directory.mkdir()
    shutil.copy(source, directory / "first.com")
    shutil.copy(source, directory / "second.com")
    server = Server(
        "test", SCHEDULER="SLURM", NUM_CORES=64, MEM_GB=192, NUM_GPUS=0
    )
    monkeypatch.chdir(tmp_path)
    with (
        patch("chemsmart.cli.sub.Server.current", return_value=server),
        patch.object(server, "submit_batch_job") as submit,
    ):
        result = CliRunner().invoke(
            sub,
            [
                "-N",
                "2",
                "--test",
                "gaussian",
                "-p",
                "test",
                "-f",
                str(directory),
                "-c",
                "0",
                "-m",
                "1",
                "modred",
                "-c",
                "[[1,2]]",
            ],
        )
    assert result.exit_code == 0, (result.output, result.exception)
    job = submit.call_args.kwargs["jobs"][0]
    args = submit.call_args.kwargs["cli_args"][0]
    assert args[args.index("--filename") + 1].endswith("first.com")
    captured = []

    def capture(actual, runner):
        captured.append(actual)
        actual.is_complete = lambda: True

    with (
        patch("chemsmart.cli.run.JobRunner") as runner,
        patch("chemsmart.cli.run._run_single_job", side_effect=capture),
    ):
        runner.return_value = job.jobrunner
        replay = CliRunner().invoke(run, args)
    assert replay.exit_code == 0, (replay.output, replay.exception)
    assert captured[0].label == job.label
    assert captured[0].settings.charge == 0
    assert captured[0].settings.multiplicity == 1
    replay_settings = captured[0].settings
    captured.clear()
    with (
        patch("chemsmart.cli.run.JobRunner") as runner,
        patch("chemsmart.cli.run._run_single_job", side_effect=capture),
    ):
        runner.return_value = job.jobrunner
        direct = CliRunner().invoke(run, args[2:])
    assert direct.exit_code == 0, (direct.output, direct.exception)
    assert replay_settings.functional == captured[0].settings.functional
    assert replay_settings.basis == captured[0].settings.basis
    assert replay_settings.modred == captured[0].settings.modred
