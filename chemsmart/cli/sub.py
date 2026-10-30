"""
Submission of jobs to queuing system via cli.

This module provides command-line interface for submitting jobs to
various queuing systems and cluster schedulers.
"""

import getpass
import logging
import os
import subprocess
import time

import click

from chemsmart.cli.jobrunner import click_jobrunner_options
from chemsmart.cli.logger import logger_options
from chemsmart.cli.subcommands import subcommands
from chemsmart.jobs.runner import JobRunner
from chemsmart.settings.server import Server
from chemsmart.utils.cli import CtxObjArguments, MyGroup
from chemsmart.utils.logger import create_logger

logger = logging.getLogger(__name__)


@click.group(name="sub", cls=MyGroup)
@click.pass_context
@click_jobrunner_options
@logger_options
@click.option(
    "-t",
    "--time-hours",
    type=float,
    default=None,
    help="Time limit in hours for the job (e.g., 48.0).",
)
@click.option("-q", "--queue", type=str, help="Queue name for job submission.")
@click.option(
    "-v/",
    "--verbose/--no-verbose",
    default=False,
    help="Turn on logging to stream output and debug logging.",
)
@click.option(
    "--test/--no-test",
    default=False,
    help="If true, job will not be submitted; only run and submit "
    "scripts will be written.",
)
@click.option(
    "--print-command/--no-print-command",
    default=False,
    help="Print the generated command.",
)
@click.option(
    "--max-parallel",
    type=click.IntRange(min=1),
    default=None,
    help="For xTB on Slurm, keep at most this many jobs queued or running. "
    "The submission process must remain running.",
)
def sub(
    ctx,
    server,
    num_cores,
    num_gpus,
    mem_gb,
    fake,
    scratch,
    delete_scratch,
    debug,
    stream,
    time_hours,
    queue,
    verbose,
    test,
    print_command,
    max_parallel,
    **kwargs,
):
    """
    Main command for submitting chemsmart jobs to queuing systems.

    This command prepares and submits jobs to cluster schedulers with
    specified resource requirements and queue parameters.
    """
    # Set up logging
    if verbose:
        create_logger(stream=True, debug=True)
    else:
        create_logger(debug=debug, stream=stream)
    logger.info("Entering main program")

    # Resolve one shared Server instance for both JobRunner and Submitter.
    server = (
        Server.current() if server is None else Server.from_servername(server)
    )
    if time_hours is not None:
        server.num_hours = time_hours
    if queue is not None:
        server.queue_name = queue
    if num_cores is not None:
        server.kwargs["NUM_CORES"] = num_cores
    if num_gpus is not None:
        server.kwargs["NUM_GPUS"] = num_gpus
    if mem_gb is not None:
        server.kwargs["MEM_GB"] = mem_gb

    jobrunner = JobRunner(
        server=server,
        scratch=scratch,
        delete_scratch=delete_scratch,
        fake=fake,
        num_cores=num_cores,
        num_gpus=num_gpus,
        mem_gb=mem_gb,
    )

    # Log the scratch value for debugging purposes
    logger.debug(f"Scratch value passed from CLI: {scratch}")

    # Store the jobrunner and other options in the context object
    ctx.ensure_object(dict)  # Ensure ctx.obj is initialized as a dict
    ctx.obj["jobrunner"] = jobrunner
    ctx.obj["max_parallel"] = max_parallel


@sub.result_callback(replace=True)
@click.pass_context
def process_pipeline(ctx, *args, **kwargs):  # noqa: PLR0915
    """
    Process the job for submission to queuing system.

    This callback function handles job submission by reconstructing
    command-line arguments and interfacing with the appropriate
    scheduler system.
    """

    def _clean_command_list(commands):
        """
        Remove keywords used in sub.py but not in run.py from a command list.

        Specifically: Some keywords/options (like queue, verbose, etc.)
        are only relevant to sub.py and not applicable to run.py.
        """
        # Get "sub" command and assert that there is exactly one.
        command = next(
            (
                subcommand
                for subcommand in commands
                if subcommand["name"] == "sub"
            ),
            None,
        )

        if command:
            # Find the keywords that are valid in sub.py
            # but should not be passed to run.py and remove those
            keywords_not_in_run = [
                "time_hours",
                "queue",
                "verbose",
                "test",
                "print_command",
                "max_parallel",
            ]

            for keyword in keywords_not_in_run:
                # Remove keyword if it exists
                command["kwargs"].pop(keyword, None)

    def _clean_command(ctx):
        """
        Remove keywords used in sub.py but not in run.py.
        """
        if "subcommand" in ctx.obj:
            _clean_command_list(ctx.obj["subcommand"])
        return ctx

    def _replace_batch_table_tokens(cli_args, batch_entry):
        """Rewrite table-mode args to a single-entry invocation.

        For pKa table mode we generated one job per table row. Submission
        scripts should therefore execute just that row, not the entire table.
        """
        if not batch_entry:
            return cli_args

        args = list(cli_args)
        table_option_names = {
            "-f",
            "--filename",
        }

        # Replace table filename with row filepath.
        for idx in range(len(args) - 1):
            if args[idx] in table_option_names:
                args[idx + 1] = str(batch_entry["filepath"])
                break

        # Per-row scripts should run a single pKa submission, not table mode.
        if "batch" in args:
            args[args.index("batch")] = "submit"

        def _drop_option(tokens, option_names):
            idx = 0
            while idx < len(tokens):
                if tokens[idx] in option_names:
                    del tokens[idx]
                    if idx < len(tokens):
                        del tokens[idx]
                    continue
                idx += 1

        # Ensure row-level proton/charge/multiplicity are explicit.
        option_map = {
            "--proton-index": str(batch_entry["proton_index"]),
            "-pi": str(batch_entry["proton_index"]),
            "--charge": str(batch_entry["charge"]),
            "-c": str(batch_entry["charge"]),
            "--multiplicity": str(batch_entry["multiplicity"]),
            "-m": str(batch_entry["multiplicity"]),
        }

        fragment_index = batch_entry.get("fragment_index")
        if fragment_index is not None:
            option_map["--index"] = str(fragment_index)
            option_map["-i"] = str(fragment_index)

        batch_label = batch_entry.get("label")
        if batch_label is not None:
            option_map["--label"] = str(batch_label)
            option_map["-l"] = str(batch_label)

        def _drop_option_pair(tokens, option_names):
            idx = 0
            while idx < len(tokens):
                if tokens[idx] in option_names:
                    del tokens[idx : idx + 2]
                    continue
                idx += 1

        def _set_option(tokens, long_opt, short_opt, insert_before=None):
            if long_opt in tokens:
                pos = tokens.index(long_opt)
                if pos + 1 < len(tokens):
                    tokens[pos + 1] = option_map[long_opt]
                return
            if short_opt in tokens:
                pos = tokens.index(short_opt)
                if pos + 1 < len(tokens):
                    tokens[pos + 1] = option_map[short_opt]
                return

            insert_idx = len(tokens)
            if insert_before in tokens:
                insert_idx = tokens.index(insert_before)
            # Prefer long-form options to avoid Click treating multi-char
            # aliases such as "-pi" as grouped short flags (e.g. "-p -i").
            tokens[insert_idx:insert_idx] = [long_opt, option_map[long_opt]]

        def _set_option_after(tokens, long_opt, short_opt, insert_after=None):
            _drop_option_pair(tokens, {long_opt, short_opt})
            insert_idx = len(tokens)
            if insert_after in tokens:
                insert_idx = tokens.index(insert_after) + 1
            tokens[insert_idx:insert_idx] = [long_opt, option_map[long_opt]]

        # Proton index is declared on ``pka submit``; place it after ``submit``
        # so per-row ``chemsmart run`` scripts parse it reliably.
        _set_option_after(args, "--proton-index", "-pi", insert_after="submit")
        # Charge/multiplicity belong to the backend group (gaussian/orca), so
        # they must appear before entering the "pka" group.
        _set_option(args, "--charge", "-c", insert_before="pka")
        _set_option(args, "--multiplicity", "-m", insert_before="pka")
        if fragment_index is not None:
            _set_option(args, "--index", "-i", insert_before="pka")
        if batch_label is not None:
            _set_option(args, "--label", "-l", insert_before="pka")

        batch_scheme = batch_entry.get("scheme")
        if batch_scheme is not None:
            option_map.update(
                {
                    "--scheme": str(batch_scheme),
                    "-s": str(batch_scheme),
                }
            )
            _set_option(args, "--scheme", "-s", insert_before="submit")

            if batch_scheme == "direct":
                # Subsequent batch rows run direct mode only; drop all
                # reference-acid CLI options so click validation does not fail.
                _drop_option(
                    args,
                    {
                        "--reference",
                        "-r",
                        "--reference-proton-index",
                        "-rpi",
                        "--reference-color-code",
                        "-rcc",
                        "--reference-charge",
                        "-rc",
                        "--reference-multiplicity",
                        "-rm",
                        "--reference-conjugate-base-charge",
                        "--reference-conjugate-base-multiplicity",
                    },
                )

        return args

    def _reconstruct_cli_args(ctx, job):
        """
        Get cli args that reconstruct the command line.

        Rebuilds the command-line arguments from the context object
        for job submission purposes.
        """
        commands = ctx.obj["subcommand"]
        args = CtxObjArguments(commands, entry_point="sub")
        cli_args = args.reconstruct_command_line()[
            1:
        ]  # remove the first element 'sub'
        cli_args = _replace_batch_table_tokens(
            cli_args, getattr(job, "_batch_entry", None)
        )

        if kwargs.get("print_command"):
            print(cli_args)
        return cli_args

    def _process_single_job(job):
        if kwargs.get("test"):
            logger.warning('Not submitting as "test" flag specified.')

        cli_args = _reconstruct_cli_args(ctx, job)

        jobrunner.server.submit(
            job=job,
            test=kwargs.get("test"),
            cli_args=cli_args,
        )

    ctx = _clean_command(ctx)
    jobrunner = ctx.obj["jobrunner"]
    job = args[0]

    max_parallel = ctx.obj.get("max_parallel")
    if max_parallel is not None:
        if jobrunner.server.scheduler != "SLURM":
            raise click.UsageError("--max-parallel requires a Slurm server")
        if not (ctx.obj.get("filename") or "").lower().endswith(".npz"):
            raise click.UsageError(
                "--max-parallel currently requires an NPZ input"
            )
        if not isinstance(job, list) or not all(
            j.PROGRAM == "xtb" for j in job
        ):
            raise click.UsageError(
                "--max-parallel requires multiple xTB structures selected with -i"
            )
        indices = ctx.obj.get("molecule_indices")
        if indices is None or len(indices) != len(job):
            raise click.UsageError(
                "A separate structure index is required for each xTB job"
            )
        if kwargs.get("test"):
            raise click.UsageError(
                "--test cannot be combined with --max-parallel"
            )

        cli_args = _reconstruct_cli_args(ctx, job[0])
        submit_limited_xtb_jobs(
            jobrunner, job, indices, cli_args, max_parallel
        )
        return

    # Handle list of jobs (when multiple molecules are specified with --index)
    if isinstance(job, list):
        logger.info(f"Processing {len(job)} jobs")
        for single_job in job:
            single_job.jobrunner = jobrunner
            _process_single_job(job=single_job)
    else:
        job.jobrunner = jobrunner
        _process_single_job(job=job)


def _xtb_single_args(cli_args, index, label, source_xyz=None):
    """Build a run invocation selecting and naming exactly one structure."""
    args = list(cli_args)
    pos = args.index("xtb") + 1
    end = next(
        (i for i in range(pos, len(args)) if args[i] in {"opt", "sp", "hess"}),
        len(args),
    )
    section = args[pos:end]
    for option in (
        "--index",
        "-i",
        "--structure-index",
        "--si",
        "--label",
        "-l",
        "--append-label",
        "-a",
    ):
        while option in section:
            i = section.index(option)
            del section[i : i + 2]
    if source_xyz is not None:
        for option in ("--filename", "-f"):
            if option in section:
                section[section.index(option) + 1] = source_xyz
                break
        else:
            raise ValueError("Expected an xTB input filename")
        selection = []
    else:
        selection = ["--index", str(index)]
    args[pos:end] = section + selection + ["--label", label]
    return args


def submit_limited_xtb_jobs(
    jobrunner, jobs, indices, cli_args, limit, poll_interval=30
):
    """Submit at most limit Slurm jobs, replenishing as jobs leave the queue.

    The controller runs in the foreground; scheduler query and submission
    failures stop it rather than risking exceeding the limit.
    """
    active = set()
    server = jobrunner.server
    for job, index in zip(jobs, indices):
        job.jobrunner = jobrunner
        if job.skip_completed and job.is_complete():
            continue
        while len(active) >= limit:
            result = subprocess.run(
                [
                    "squeue",
                    "--noheader",
                    "--user",
                    getpass.getuser(),
                    "--format=%i",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            active.intersection_update(result.stdout.split())
            if len(active) >= limit:
                time.sleep(poll_interval)
        server._check_running_jobs(job)
        # Materialize the already-read geometry once. Each compute job reads
        # its own small input instead of reopening the entire NPZ archive.
        source_xyz = os.path.join(job.folder, f"{job.label}.source.xyz")
        job.molecule.write_xyz(source_xyz, mode="w")
        server._write_submission_script(
            job, _xtb_single_args(cli_args, index, job.label, source_xyz)
        )
        submitter = server.get_submitter(job)
        result = subprocess.run(
            ["sbatch", "--parsable", submitter.submit_script],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"Unexpected sbatch output: {result.stdout!r}")
        active.add(job_id)
        logger.info("Submitted structure %s as Slurm job %s", index, job_id)


for subcommand in subcommands:
    sub.add_command(subcommand)
