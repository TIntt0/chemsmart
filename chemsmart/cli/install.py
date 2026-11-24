"""Install optional CHEMSMART plugins in the current Python environment."""

import subprocess
import sys

import click


@click.group()
def install():
    """Install plugins and optional model backends."""


@install.command()
@click.argument(
    "backend", required=False, type=click.Choice(["mace", "sisso"])
)
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Install from a local CHEMSMART-ML source directory.",
)
@click.option(
    "--upgrade", is_flag=True, help="Upgrade an existing installation."
)
def ml(backend, source, upgrade):
    """Install CHEMSMART-ML with common models and descriptors.

    Common models, Morgan and SOAP dependencies are installed by default.
    Optionally select mace or sisso for specialized model dependencies.
    SISSO also requires a separately configured SISSO executable.

    Examples: chemsmart install ml; chemsmart install ml mace
    """
    requirement = source or "chemsmart-ml"
    if backend:
        requirement += f"[{backend}]"
    command = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    command.append(requirement)
    click.echo(f"Installing {requirement} using {sys.executable}")
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(
            "Installation failed. See the pip output above for details."
        ) from exc
    if backend == "sisso":
        click.echo(
            "Python dependencies installed. Training also requires the "
            "SISSO executable on PATH."
        )
    click.echo("Next: chemsmart run ml train --help")
