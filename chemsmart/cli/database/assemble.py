import logging
import os

import click

from chemsmart.cli.job import click_folder_options
from chemsmart.database.assemble import (
    SingleFileAssembler,
    SingleFolderAssembler,
)
from chemsmart.database.database import Database
from chemsmart.io.xtb.folder import XTBFolder
from chemsmart.jobs.job import Job
from chemsmart.utils.cli import MyCommand
from chemsmart.utils.io import get_program_type_from_file

from .database import database

logger = logging.getLogger(__name__)


class DatabaseAssemblyJob(Job):
    """Submission metadata for a database assembly command."""

    PROGRAM = "database"

    def __init__(self, output, jobrunner):
        super().__init__(
            molecule=None,
            label=os.path.splitext(os.path.basename(output))[0],
            jobrunner=jobrunner,
            skip_completed=False,
        )

    def is_complete(self):
        return False


@database.command(cls=MyCommand)
@click_folder_options
@click.option(
    "-i",
    "--index",
    default=":",
    show_default=True,
    help="Index (1-based) of the structures to extract from multi-structure files.",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default="database.db",
    show_default=True,
    help="Output database file (.db extension). ",
)
@click.option(
    "--include-failed",
    is_flag=True,
    default=False,
    help="Include records from calculations that did not terminate normally. "
    "Partial data (e.g. intermediate geometries) will be assembled when available.",
)
@click.pass_context
def assemble(
    ctx,
    directory,
    filetype,
    program,
    index,
    output,
    include_failed,
):
    """Assemble calculation output files into a chemsmart database.

    This command collects calculation data from output files in the specified
    directory and assembles them into a unified chemsmart database (.db).

    \b
    Examples:
        chemsmart run database assemble -d results/ -o my.db
        chemsmart run database assemble -d ./ -p gaussian -o database.db
    """

    # Ensure the output filename ends with .db
    if not output.lower().endswith(".db"):
        output = output + ".db"

    if directory is None:
        directory = "./"
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # A queued invocation must not discover or parse files on the submit host.
    if ctx.find_root().invoked_subcommand == "sub":
        output = os.path.abspath(output)
        for command in ctx.obj["subcommand"]:
            if command["name"] == "assemble":
                command["kwargs"]["directory"]["value"] = directory
                command["kwargs"]["output"]["value"] = output
        return DatabaseAssemblyJob(output, ctx.obj["jobrunner"])

    supported_programs = {"gaussian", "orca", "xtb"}
    if program is None:
        programs = supported_programs
    elif program.lower() in supported_programs:
        programs = {program.lower()}
    else:
        raise ValueError(
            f"Unsupported program '{program}'. "
            "Use 'gaussian', 'orca', or 'xtb'."
        )

    def iter_targets():
        """Discover and assemble supported calculations during one tree walk."""
        visited = 0
        detected = 0
        logger.info("Scanning calculation outputs in %s", directory)

        def report_error(error):
            logger.error("Cannot scan directory: %s", error)

        for folder, _dirs, filenames in os.walk(
            directory, onerror=report_error
        ):
            visited += 1
            if visited % 100 == 0:
                logger.info(
                    "Scanning: %d directories visited, %d sources found",
                    visited,
                    detected,
                )
            xtb_candidate = False
            for name in sorted(filenames):
                if name.endswith(".out"):
                    xtb_candidate = True
                if not (programs & {"gaussian", "orca"}) or not name.endswith(
                    (".log", ".out")
                ):
                    continue
                path = os.path.join(folder, name)
                if get_program_type_from_file(path) in programs & {
                    "gaussian",
                    "orca",
                }:
                    detected += 1
                    yield path, SingleFileAssembler, "filename"
            if "xtb" in programs and xtb_candidate:
                try:
                    valid = XTBFolder(
                        folder=folder
                    ).is_xtb_calculation_directory
                except ValueError as exc:
                    logger.error(
                        "Skipping invalid xTB directory '%s': %s", folder, exc
                    )
                    continue
                if valid:
                    detected += 1
                    yield folder, SingleFolderAssembler, "folder"
        logger.info(
            "Finished scanning: %d directories visited, %d sources found",
            visited,
            detected,
        )

    targets = iter_targets()

    # Commit bounded batches so long jobs make progress and retain completed work.
    db = None
    batch = []
    inserted = 0
    processed = 0
    for target, assembler_class, keyword in targets:
        try:
            assembler = assembler_class(
                **{keyword: target}, index=index, include_failed=include_failed
            )
            data = assembler.assemble_data
            if data:
                if db is None:
                    db = Database(db_file=output)
                    db.create()
                batch.append(data)
        except Exception:
            logger.exception("Failed to parse %s", target)
        processed += 1
        if len(batch) >= 100:
            inserted += db.insert_records(batch)
            batch.clear()
        if processed % 100 == 0:
            logger.info(
                "Processed %d sources; wrote %d records", processed, inserted
            )

    if batch:
        inserted += db.insert_records(batch)
    if not processed:
        logger.error("No supported calculation outputs found in %s", directory)
    if not inserted:
        logger.error("No valid data inserted.")
        return None
    logger.info("Inserted %d records into database: %s", inserted, output)
    return None
