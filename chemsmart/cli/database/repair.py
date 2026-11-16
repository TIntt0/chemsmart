"""CLI for rebuilding coordinate-derived molecular database fields."""

import json
import sqlite3

import click

from chemsmart.database.repair import StructureRepairer
from chemsmart.utils.cli import MyCommand

from .database import database


@database.command(name="repair-structure", cls=MyCommand)
@click.option(
    "-f",
    "--file",
    "--database",
    "db_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--report",
    required=True,
    type=click.Path(dir_okay=False),
    help="New JSONL report path; existing files are never overwritten.",
)
@click.option(
    "--scope",
    type=click.Choice(["all", "invalid", "missing"]),
    default="all",
    show_default=True,
    help="Select all species, invalid nonempty SMILES, or missing SMILES.",
)
@click.option(
    "--limit", type=click.IntRange(min=1), help="Maximum species to inspect."
)
@click.option(
    "--apply",
    is_flag=True,
    help="Write repaired fields and audit history to the database.",
)
def repair_structure(db_file, report, scope, limit, apply):
    """Rebuild SMILES, InChI, ring, aromaticity, chirality and fragment fields.

    Dry run by default. Coordinates, energies, identities and links are retained.
    Failed or conflicting species are reported and left unchanged.
    """
    try:
        summary = StructureRepairer(db_file).run(report, apply, scope, limit)
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary))
