#!/usr/bin/env python
"""
Angew MECP benchmark validation script.

Validates that chemsmart's MECP optimizer locates the same MECP reported in
the Angew paper (Smith/Paton/Münster, *Angew. Chem. Int. Ed.* 2017, 56, 9468).

Two modes:

1. **Reference extraction** (offline, no Gaussian needed):
   Parse the Angew test-case log files to extract reference energies and
   the MECP geometry, then print a benchmark table.

2. **Validation** (after running chemsmart mecp):
   Parse chemsmart's ``*_report.log`` output and compare the converged
   MECP energies against the Angew reference values.

Usage
-----
::

    # Extract reference values from Angew logs
    python validate_angew_mecp.py --reference --logdir /path/to/test_case_angew

    # Validate chemsmart output against reference
    python validate_angew_mecp.py --validate --report /path/to/chemsmart_report.log

    # Validate and print PASS/FAIL
    python validate_angew_mecp.py --validate --report /path/to/chemsmart_report.log \\
        --logdir /path/to/test_case_angew
"""
import re
import sys
from pathlib import Path

import click

# --- Reference values from Angew test case logs ---
# These are the SCF Done energies from the provided log files.
REFERENCE_ENERGIES = {
    "35_1A_singlet_opt_final": {
        "energy": -614.905094,
        "method": "M062X/def2TZVP",
        "description": "Singlet A optimized geometry (final SCF)",
        "source": "35_1A.log (last SCF Done before Optimization completed)",
    },
    "35_3A_triplet_opt_final": {
        "energy": -614.802221,
        "method": "M062X/def2TZVP",
        "description": "Triplet A optimized geometry (final SCF)",
        "source": "35_3A.log (last SCF Done before Optimization completed)",
    },
    "35_MECP_A_singlet_sp_gas": {
        "energy": -614.106793,
        "method": "M062X/def2SVP",
        "description": "MECP geometry, singlet SP, gas phase",
        "source": "35_MECP_A_sp_gas_phase.log",
    },
    "35_MECP_A_triplet_sp_gas": {
        "energy": -614.106710,
        "method": "M062X/def2SVP",
        "description": "MECP geometry, triplet SP, gas phase",
        "source": "35_MECP_A_triplet_gas_phase.log",
    },
    "35_MECP_A_singlet_sp_smd": {
        "energy": -614.808964,
        "method": "M062X/def2TZVP + SMD(1,4-dioxane)",
        "description": "MECP geometry, singlet SP, SMD 1,4-dioxane",
        "source": "35_MECP_A_sp_smd_1_4_dioxane.log",
    },
}

# Convergence target: at MECP, |E_A - E_B| should be < ~1e-4 Hartree
# (0.06 kcal/mol). The Angew reference shows ΔE = 0.052 kcal/mol.
REFERENCE_DELTA_E_KCAL = 0.052  # |E_A - E_B| at MECP, gas phase, def2SVP

# Tolerances for PASS/FAIL
TOL_DELTA_E_KCAL = 1.0  # |ΔE| at MECP must be < 1.0 kcal/mol
TOL_ENERGY_MATCH_KCAL = 2.0  # chemsmart MECP energy must match ref within 2 kcal/mol

_HARTREE_TO_KCAL = 627.509474

_SCF_RE = re.compile(r"SCF Done:\s+E\([RU]\w+\)\s+=\s+([+-]?\d+\.?\d*)")
_OPT_DONE_RE = re.compile(r"Optimization completed")


def extract_reference_from_logs(logdir):
    """Parse Angew log files and print reference values."""
    logdir = Path(logdir)

    print("=" * 80)
    print("Angew MECP Reference Values (extracted from log files)")
    print("=" * 80)

    files = {
        "35_1A.log": "Singlet A optimization",
        "35_3A.log": "Triplet A optimization",
        "35_MECP_A_sp_gas_phase.log": "MECP singlet SP (gas, def2SVP)",
        "35_MECP_A_triplet_gas_phase.log": "MECP triplet SP (gas, def2SVP)",
        "35_MECP_A_sp_smd_1_4_dioxane.log": "MECP singlet SP (SMD, def2TZVP)",
    }

    for fname, desc in files.items():
        fpath = logdir / fname
        if not fpath.exists():
            print(f"  [MISSING] {fname} ({desc})")
            continue

        with open(fpath, encoding="utf-8", errors="replace") as f:
            content = f.read()

        scf_matches = _SCF_RE.findall(content)
        opt_done = _OPT_DONE_RE.search(content)

        if scf_matches:
            # For opt logs: take last SCF before "Optimization completed"
            # For SP logs: take the only SCF
            last_energy = float(scf_matches[-1])
            n_scf = len(scf_matches)
            status = "OPTIMIZED" if opt_done else "SINGLE POINT"
            print(
                f"  {fname:45s} {desc:40s}\n"
                f"    -> {status}, {n_scf} SCF cycle(s), "
                f"E = {last_energy:.6f} Ha "
                f"({last_energy * _HARTREE_TO_KCAL:.2f} kcal/mol)"
            )
        else:
            print(f"  [NO SCF] {fname} ({desc})")

    # Compute MECP delta E
    print("\n" + "-" * 80)
    print("MECP validation (gas phase, def2SVP):")
    e_a = REFERENCE_ENERGIES["35_MECP_A_singlet_sp_gas"]["energy"]
    e_b = REFERENCE_ENERGIES["35_MECP_A_triplet_sp_gas"]["energy"]
    dE = abs(e_a - e_b) * _HARTREE_TO_KCAL
    print(f"  E_A (singlet)  = {e_a:.6f} Ha")
    print(f"  E_B (triplet)  = {e_b:.6f} Ha")
    print(f"  |E_A - E_B|    = {dE:.4f} kcal/mol  (target: < {TOL_DELTA_E_KCAL})")
    print(f"  STATUS: {'PASS' if dE < TOL_DELTA_E_KCAL else 'FAIL'}")

    # Solvent correction
    print("\n" + "-" * 80)
    print("Solvent correction (SMD, 1,4-dioxane, def2TZVP):")
    e_solv = REFERENCE_ENERGIES["35_MECP_A_singlet_sp_smd"]["energy"]
    print(f"  E (SMD)  = {e_solv:.6f} Ha")
    print(f"  (Note: paper uses EtOAc; log uses 1,4-dioxane)")

    print("=" * 80)


def validate_chemsmart_report(report_path, logdir=None):
    """Parse chemsmart *_report.log and compare against reference."""
    report_path = Path(report_path)

    if not report_path.exists():
        print(f"ERROR: Report file not found: {report_path}")
        sys.exit(1)

    with open(report_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Parse report
    step_re = re.compile(
        r"step=(\d+)\s+"
        r"E_A=([+-]?\d+\.?\d*)\s+"
        r"E_B=([+-]?\d+\.?\d*)\s+"
        r"dE=([+-]?\d+\.?\d*[eE]?[+-]?\d*)"
    )
    converged_re = re.compile(r"Converged at step (\d+)\.")

    last_step = None
    converged = False
    for line in lines:
        m = step_re.search(line)
        if m:
            last_step = m
        m2 = converged_re.search(line)
        if m2:
            converged = True

    if last_step is None:
        print("ERROR: No step data found in report.")
        sys.exit(1)

    g = last_step.groups()
    step_n = int(g[0])
    e_a = float(g[1])
    e_b = float(g[2])
    dE_hartree = float(g[3])
    dE_kcal = abs(dE_hartree) * _HARTREE_TO_KCAL

    print("=" * 80)
    print(f"Chemsmart MECP Report Validation: {report_path.name}")
    print("=" * 80)
    print(f"  Converged:    {'YES' if converged else 'NO'}")
    print(f"  Final step:   {step_n}")
    print(f"  E_A:          {e_a:.6f} Ha")
    print(f"  E_B:          {e_b:.6f} Ha")
    print(f"  |E_A - E_B|:  {dE_kcal:.4f} kcal/mol")

    # Check 1: |E_A - E_B| at MECP
    check1_pass = dE_kcal < TOL_DELTA_E_KCAL
    print(f"\n  Check 1: |ΔE| < {TOL_DELTA_E_KCAL} kcal/mol")
    print(f"    -> {'PASS' if check1_pass else 'FAIL'} ({dE_kcal:.4f})")

    # Check 2: Energy match against reference (if logdir given)
    if logdir:
        ref_e_a = REFERENCE_ENERGIES["35_MECP_A_singlet_sp_gas"]["energy"]
        ref_e_b = REFERENCE_ENERGIES["35_MECP_A_triplet_sp_gas"]["energy"]

        dev_a = abs(e_a - ref_e_a) * _HARTREE_TO_KCAL
        dev_b = abs(e_b - ref_e_b) * _HARTREE_TO_KCAL

        check2_pass = dev_a < TOL_ENERGY_MATCH_KCAL and dev_b < TOL_ENERGY_MATCH_KCAL
        print(f"\n  Check 2: Energy match (ref E_A={ref_e_a:.6f}, ref E_B={ref_e_b:.6f})")
        print(f"    |E_A - ref_A| = {dev_a:.4f} kcal/mol")
        print(f"    |E_B - ref_B| = {dev_b:.4f} kcal/mol")
        print(f"    -> {'PASS' if check2_pass else 'FAIL'} (tol: {TOL_ENERGY_MATCH_KCAL})")

        all_pass = check1_pass and check2_pass and converged
    else:
        all_pass = check1_pass and converged

    print(f"\n  {'='*40}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {'='*40}")
    return all_pass


@click.command()
@click.option(
    "--reference",
    is_flag=True,
    help="Extract reference values from Angew log files.",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate a chemsmart *_report.log against the Angew reference.",
)
@click.option(
    "--logdir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory containing the Angew test-case log files.",
)
@click.option(
    "--report",
    type=click.Path(exists=True),
    default=None,
    help="Path to chemsmart *_report.log file to validate.",
)
def entry_point(reference, validate, logdir, report):
    """Validate chemsmart MECP output against the Angew paper reference."""

    if reference:
        if not logdir:
            click.echo("ERROR: --logdir required with --reference")
            sys.exit(1)
        extract_reference_from_logs(logdir)

    if validate:
        if not report:
            click.echo("ERROR: --report required with --validate")
            sys.exit(1)
        validate_chemsmart_report(report, logdir)

    if not reference and not validate:
        click.echo("Use --reference or --validate. See --help.")
        sys.exit(1)


if __name__ == "__main__":
    entry_point()
