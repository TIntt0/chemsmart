import json
from types import SimpleNamespace

from chemsmart.jobs.batch import _batch_capacity, run_slurm_batch


def test_batch_capacity_respects_core_and_memory_limits():
    manifest = {
        "num_nodes": 5,
        "cores_per_node": 64,
        "memory_per_node_gb": 375,
        "gpus_per_node": 0,
        "task_cores": 1,
        "task_memory_gb": 10,
        "task_gpus": 0,
    }

    assert _batch_capacity(manifest) == 185


def test_batch_runner_continues_after_task_failure(tmp_path, mocker, capsys):
    manifest_path = tmp_path / "batch.json"
    manifest_path.write_text(
        json.dumps(
            {
                "num_nodes": 2,
                "cores_per_node": 4,
                "memory_per_node_gb": 8,
                "gpus_per_node": 0,
                "task_cores": 1,
                "task_memory_gb": 2,
                "task_gpus": 0,
                "tasks": [
                    {"label": f"structure_{i}", "runscript": f"run_{i}.py"}
                    for i in range(3)
                ],
            }
        )
    )
    commands = []

    def run_task(command, check, stdout, stderr):
        commands.append(command)
        return SimpleNamespace(
            returncode=1 if command[-1] == "run_1.py" else 0
        )

    mocker.patch("chemsmart.jobs.batch.subprocess.run", side_effect=run_task)

    assert run_slurm_batch(manifest_path) == 1
    assert len(commands) == 3
    assert all("--cpus-per-task=1" in command for command in commands)

    report = json.loads((tmp_path / "batch.report.json").read_text())
    assert report["successful"] == 2
    assert report["failed"] == 1
    assert report["unfinished"] == 0
    assert [task["status"] for task in report["tasks"]] == [
        "successful",
        "failed",
        "successful",
    ]
    assert "successful=2, failed=1, unfinished=0" in capsys.readouterr().out


def test_batch_starts_next_task_before_other_task_finishes(tmp_path, mocker):
    from threading import Event

    manifest_path = tmp_path / "batch.json"
    manifest_path.write_text(
        json.dumps(
            {
                "num_nodes": 1,
                "cores_per_node": 2,
                "memory_per_node_gb": 2,
                "gpus_per_node": 0,
                "task_cores": 1,
                "task_memory_gb": 1,
                "task_gpus": 0,
                "tasks": [
                    {"label": str(i), "runscript": f"run_{i}.py"}
                    for i in range(3)
                ],
            }
        )
    )
    first_started = Event()
    third_started = Event()

    def run_task(command, check, stdout, stderr):
        if command[-1] == "run_0.py":
            first_started.set()
            assert third_started.wait(
                timeout=5
            ), "next task did not start while first was running"
        elif command[-1] == "run_1.py":
            assert first_started.wait(timeout=5)
        elif command[-1] == "run_2.py":
            third_started.set()
        return SimpleNamespace(returncode=0)

    mocker.patch("chemsmart.jobs.batch.subprocess.run", side_effect=run_task)
    assert run_slurm_batch(manifest_path) == 0
    assert third_started.is_set()
