import functools
import logging
import os

import click

from chemsmart.jobs.job import Job
from chemsmart.utils.cli import MyGroup

logger = logging.getLogger(__name__)


@click.group(name="database", cls=MyGroup)
@click.pass_context
def database(ctx):
    """CLI for chemsmart database operations."""
    pass


class DatabaseJob(Job):
    """Metadata for a database command submitted to a scheduler."""

    PROGRAM = "database"

    def __init__(self, label, jobrunner):
        super().__init__(
            molecule=None,
            label=label,
            jobrunner=jobrunner,
            skip_completed=False,
        )

    def is_complete(self):
        return False


def prepare_database_submission(ctx, operation, paths):
    """Return a job for sub; normalize paths before reconstructing the command."""
    if ctx.find_root().invoked_subcommand != "sub":
        return None
    command = next(
        item for item in ctx.obj["subcommand"] if item["name"] == operation
    )
    for name, value in paths.items():
        if value is not None:
            command["kwargs"][name]["value"] = os.path.abspath(value)
    source = paths.get("output") or paths.get("file") or "database"
    label = f"{operation}_{os.path.splitext(os.path.basename(source))[0]}"
    return DatabaseJob(label, ctx.obj["jobrunner"])


def click_database_id_options(f):
    """CLI options for selecting records/molecules/structures from a chemsmart database
    by ID or index. Only relevant when the input file (-f) is a chemsmart database (.db).
    """

    @click.option(
        "--ri",
        "--record-index",
        "record_index",
        type=int,
        default=None,
        help="Record index (1-based) within a chemsmart database.",
    )
    @click.option(
        "--rid",
        "--record-id",
        "record_id",
        type=str,
        default=None,
        help="Record ID (or unique prefix) within a chemsmart database.",
    )
    @click.option(
        "--si",
        "--structure-index",
        "structure_index",
        type=str,
        default=None,
        help="Structure index (1-based) within the selected record.",
    )
    @click.option(
        "--sid",
        "--structure-id",
        "structure_id",
        type=str,
        default=None,
        help="Global structure ID (or unique prefix) within a chemsmart database.",
    )
    @click.option(
        "--mid",
        "--molecule-id",
        "molecule_id",
        type=str,
        default=None,
        help="Molecule ID (or unique prefix); selects every conformer of that molecule.",
    )
    @functools.wraps(f)
    def wrapper_common_options(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper_common_options
