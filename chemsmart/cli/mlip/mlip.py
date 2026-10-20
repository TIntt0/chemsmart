"""Run MACE machine-learned potentials through ASE."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE, LBFGS

from chemsmart.calculators.mlip import MLIPCalculator
from chemsmart.utils.cli import MyCommand, MyGroup


def _output_path(output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def _build_calculator(config: dict):
    try:
        return MLIPCalculator(
            backend=config["backend"],
            pretrained=config["pretrained"],
            checkpoint=config["checkpoint"],
            device=config["device"],
            default_dtype=config["default_dtype"],
            dispersion=config["dispersion"],
        ).build()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _load_atoms(config: dict):
    atoms = read(config["filename"], index=config["index"])
    if isinstance(atoms, list):
        raise click.ClickException(
            "The selected input contains multiple structures; choose one "
            "with --index."
        )
    atoms.calc = _build_calculator(config)
    return atoms


@click.group(cls=MyGroup)
@click.pass_context
@click.option(
    "-f",
    "--filename",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input structure readable by ASE (for example XYZ, CIF, or POSCAR).",
)
@click.option("--index", default=-1, show_default=True, type=int)
@click.option(
    "--backend",
    type=click.Choice(["mace"], case_sensitive=False),
    default="mace",
    show_default=True,
)
@click.option("--pretrained", type=str, default=None)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--default-dtype",
    type=click.Choice(["float32", "float64"]),
    default="float64",
    show_default=True,
)
@click.option("--dispersion/--no-dispersion", default=False, show_default=True)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
def mlip(ctx, **kwargs):
    """Run MLIP calculations through the ASE calculator interface."""
    if kwargs["pretrained"] is not None and kwargs["checkpoint"] is not None:
        raise click.UsageError(
            "--pretrained and --checkpoint are mutually exclusive."
        )
    ctx.ensure_object(dict)
    ctx.obj["mlip"] = kwargs


@mlip.command("sp", cls=MyCommand)
@click.pass_context
def single_point(ctx):
    """Calculate potential energy and atomic forces."""
    config = ctx.obj["mlip"]
    atoms = _load_atoms(config)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    result = {
        "backend": config["backend"],
        "energy_eV": energy,
        "forces_eV_per_Angstrom": forces.tolist(),
    }
    output = _output_path(config["output_dir"], "mlip_sp.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    click.echo(f"Energy: {energy:.12f} eV")
    click.echo(f"Results: {output}")


@mlip.command("opt", cls=MyCommand)
@click.pass_context
@click.option(
    "--fmax",
    type=click.FloatRange(min=0, min_open=True),
    default=0.05,
    show_default=True,
)
@click.option(
    "--steps", type=click.IntRange(min=1), default=500, show_default=True
)
@click.option(
    "--optimizer",
    type=click.Choice(["bfgs", "lbfgs", "fire"]),
    default="bfgs",
    show_default=True,
)
def optimize(ctx, fmax, steps, optimizer):
    """Optimize a structure with MLIP energies and forces."""
    config = ctx.obj["mlip"]
    atoms = _load_atoms(config)
    output_dir = config["output_dir"]
    trajectory = _output_path(output_dir, "mlip_opt.traj")
    logfile = _output_path(output_dir, "mlip_opt.log")
    optimizer_cls = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE}[optimizer]
    dynamics = optimizer_cls(
        atoms, trajectory=str(trajectory), logfile=str(logfile)
    )
    converged = bool(dynamics.run(fmax=fmax, steps=steps))
    structure = _output_path(output_dir, "mlip_opt.xyz")
    write(structure, atoms)
    summary = {
        "converged": converged,
        "steps": int(dynamics.get_number_of_steps()),
        "energy_eV": float(atoms.get_potential_energy()),
        "max_force_eV_per_Angstrom": float(
            np.linalg.norm(atoms.get_forces(), axis=1).max()
        ),
    }
    result = _output_path(output_dir, "mlip_opt.json")
    result.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    click.echo(f"Converged: {converged}")
    click.echo(f"Optimized structure: {structure}")


@mlip.command("md", cls=MyCommand)
@click.pass_context
@click.option(
    "--steps", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option(
    "--timestep-fs",
    type=click.FloatRange(min=0, min_open=True),
    default=0.5,
    show_default=True,
)
@click.option(
    "--temperature-k",
    type=click.FloatRange(min=0),
    default=300.0,
    show_default=True,
)
@click.option(
    "--sample-interval",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
)
@click.option("--seed", type=int, default=None)
def molecular_dynamics(
    ctx,
    steps,
    timestep_fs,
    temperature_k,
    sample_interval,
    seed,
):
    """Run basic NVE molecular dynamics with Velocity Verlet."""
    config = ctx.obj["mlip"]
    atoms = _load_atoms(config)
    rng = np.random.RandomState(seed) if seed is not None else None
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    dynamics = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    trajectory = _output_path(config["output_dir"], "mlip_md.traj")
    energies = _output_path(config["output_dir"], "mlip_md.csv")
    with energies.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "time_fs",
                "potential_eV",
                "kinetic_eV",
                "total_eV",
                "temperature_K",
            ]
        )

        def sample():
            potential = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            writer.writerow(
                [
                    dynamics.nsteps,
                    dynamics.get_time() / units.fs,
                    potential,
                    kinetic,
                    potential + kinetic,
                    atoms.get_temperature(),
                ]
            )
            handle.flush()

        dynamics.attach(sample, interval=sample_interval)
        dynamics.attach(
            write,
            interval=sample_interval,
            filename=trajectory,
            images=atoms,
            append=True,
        )
        dynamics.run(steps)
    final_structure = _output_path(config["output_dir"], "mlip_md_final.xyz")
    write(final_structure, atoms)
    click.echo(f"Trajectory: {trajectory}")
    click.echo(f"Energies: {energies}")
