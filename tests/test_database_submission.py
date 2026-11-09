from importlib import import_module
from unittest.mock import patch

from click.testing import CliRunner

from chemsmart.cli.main import entry_point
from chemsmart.settings.server import Server
from chemsmart.settings.submitters import SLURMSubmitter


def test_database_assembly_submission_defers_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = Server(name="dummy")
    with (
        patch.object(Server, "from_servername", return_value=server),
        patch.object(server, "submit") as submit,
        patch.object(
            import_module("chemsmart.cli.database.assemble").os, "walk"
        ) as folder,
    ):
        result = CliRunner().invoke(
            entry_point,
            [
                "sub",
                "-s",
                "dummy",
                "--test",
                "database",
                "assemble",
                "-d",
                ".",
                "-p",
                "xtb",
                "-o",
                "xtb.db",
            ],
        )
    assert result.exit_code == 0, result.output + repr(result.exception)
    folder.assert_not_called()
    submit.assert_called_once()
    args = submit.call_args.kwargs["cli_args"]
    assert str(tmp_path) in args
    assert str(tmp_path / "xtb.db") in args
    assert not (tmp_path / "xtb.db").exists()


def test_database_submission_writes_scripts_without_executable_config(
    tmp_path, monkeypatch, slurm_test_server
):
    from chemsmart.cli.database.assemble import DatabaseAssemblyJob

    monkeypatch.chdir(tmp_path)
    job = DatabaseAssemblyJob("xtb.db", None)
    submitter = SLURMSubmitter(job=job, server=slurm_test_server)
    submitter.write(["database", "assemble", "-d", str(tmp_path), "-p", "xtb"])
    assert (tmp_path / submitter.run_script).exists()
    assert (tmp_path / submitter.submit_script).exists()


def test_assembly_writes_bounded_batches(tmp_path):
    from unittest.mock import MagicMock

    from chemsmart.cli.run import run

    module = import_module("chemsmart.cli.database.assemble")

    def walk(_directory, onerror=None):
        for i in range(205):
            yield str(tmp_path / str(i)), [], ["test.log"]

    assembler = MagicMock()
    assembler.assemble_data = {"record": "example"}
    database = MagicMock()
    batches = []
    database.insert_records.side_effect = lambda records: (
        batches.append(len(records)) or len(records)
    )
    with (
        patch.object(module.os, "walk", side_effect=walk),
        patch.object(
            module, "get_program_type_from_file", return_value="gaussian"
        ),
        patch.object(module, "SingleFileAssembler", return_value=assembler),
        patch.object(module, "Database", return_value=database),
    ):
        result = CliRunner().invoke(
            run,
            ["database", "assemble", "-d", str(tmp_path), "-p", "gaussian"],
        )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert batches == [100, 100, 5]


def test_discovery_reports_progress_before_parsing(tmp_path):
    from chemsmart.cli.run import run

    for i in range(105):
        (tmp_path / str(i)).mkdir()
    module = import_module("chemsmart.cli.database.assemble")
    with patch.object(module.logger, "info") as info:
        result = CliRunner().invoke(
            run, ["database", "assemble", "-d", str(tmp_path), "-p", "xtb"]
        )
    assert result.exit_code == 0, result.output
    messages = [call.args[0] % call.args[1:] for call in info.call_args_list]
    assert any("100 directories visited" in message for message in messages)
    assert any("106 directories visited" in message for message in messages)


def test_xtb_assembles_before_scan_completes(tmp_path):
    from unittest.mock import MagicMock

    from chemsmart.cli.run import run

    module = import_module("chemsmart.cli.database.assemble")
    assembler = MagicMock()
    assembler.assemble_data = {"record": "example"}
    database = MagicMock()
    database.insert_records.side_effect = lambda records: len(records)

    def walk(_directory, onerror=None):
        for i in range(100):
            yield str(tmp_path / str(i)), [], ["xtb.out"]
        assert database.insert_records.call_count == 1
        yield str(tmp_path / "last"), [], ["xtb.out"]

    with (
        patch.object(module.os, "walk", side_effect=walk),
        patch.object(module, "XTBFolder") as folder,
        patch.object(module, "SingleFolderAssembler", return_value=assembler),
        patch.object(module, "Database", return_value=database),
    ):
        folder.return_value.is_xtb_calculation_directory = True
        result = CliRunner().invoke(
            run, ["database", "assemble", "-d", str(tmp_path), "-p", "xtb"]
        )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert database.insert_records.call_count == 2


def test_real_xtb_folder_assembles_to_database(tmp_path, xtb_water_outfolder):
    from chemsmart.cli.run import run
    from chemsmart.database.database import Database

    path = tmp_path / "xtb.db"
    result = CliRunner().invoke(
        run,
        [
            "database",
            "assemble",
            "-d",
            xtb_water_outfolder,
            "-p",
            "xtb",
            "-o",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert Database(str(path)).count_records() == 1
