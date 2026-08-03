#!/usr/bin/env python
"""
MECP benchmark results aggregator.

Scans a directory for chemsmart MECP ``*_report.log`` files, extracts
convergence status, final energies, and step count for each job, and
writes a summary table (CSV and/or markdown).

Designed for combinatorial MECP benchmarking where jobs are labelled
``{base}_{functional}_{basis}`` (and optionally ``_idx{N}``).

Usage
-----
::

    chemsmart-mecp-summary -d /path/to/jobs -o results.csv
    chemsmart-mecp-summary -d /path/to/jobs -o results.csv --markdown results.md
    chemsmart-mecp-summary -d /path/to/jobs  # prints to stdout only
"""
import csv
import logging
import os
import re
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# step=000001 E_A=-614.106793 E_B=-614.106710 dE=...
_STEP_RE = re.compile(
    r"step=(\d+)\s+"
    r"E_A=([+-]?\d+\.?\d*)\s+"
    r"E_B=([+-]?\d+\.?\d*)\s+"
    r"dE=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"pgrad_max=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"pgrad_rms=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"disp_max=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"disp_rms=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"seam_max=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"seam_rms=([+-]?\d+\.?\d*[eE]?[+-]?\d*)\s+"
    r"step_size=([+-]?\d+\.?\d*[eE]?[+-]?\d*)"
)

_CONVERGED_RE = re.compile(r"Converged at step (\d+)\.")

_HARTREE_TO_KCAL = 627.509474


def parse_report(filepath):
    """Parse a single ``*_report.log`` file.

    Returns a dict with keys: label, converged, n_steps, e_a, e_b, dE_hartree,
    dE_kcal, pgrad_max, pgrad_rms, disp_max, disp_rms, step_size, or None
    if the file is not a valid MECP report.
    """
    path = Path(filepath)
    label = path.name.replace("_report.log", "")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or "CHEMSMART" not in lines[0]:
        return None

    last_step = None
    converged_step = None

    for line in lines:
        m = _STEP_RE.search(line)
        if m:
            last_step = m
        m2 = _CONVERGED_RE.search(line)
        if m2:
            converged_step = int(m2.group(1))

    if last_step is None:
        return {
            "label": label,
            "converged": False,
            "n_steps": 0,
            "e_a": None,
            "e_b": None,
            "dE_hartree": None,
            "dE_kcal": None,
            "pgrad_max": None,
            "pgrad_rms": None,
            "disp_max": None,
            "disp_rms": None,
            "step_size": None,
        }

    g = last_step.groups()
    step_idx = int(g[0])
    e_a = float(g[1])
    e_b = float(g[2])
    dE = float(g[3])
    n_steps = converged_step if converged_step is not None else step_idx

    return {
        "label": label,
        "converged": converged_step is not None,
        "n_steps": n_steps,
        "e_a": e_a,
        "e_b": e_b,
        "dE_hartree": dE,
        "dE_kcal": dE * _HARTREE_TO_KCAL,
        "pgrad_max": float(g[4]),
        "pgrad_rms": float(g[5]),
        "disp_max": float(g[6]),
        "disp_rms": float(g[7]),
        "step_size": float(g[9]),
    }


def parse_label_for_combinatorial(label):
    """Try to split a combinatorial label into (base, functional, basis).

    Label format: ``{base}_{functional}_{basis}`` or
    ``{base}_{functional}_{basis}_idx{N}``.

    This is best-effort: if the label doesn't match the combinatorial
    pattern, returns (label, None, None).
    """
    # Strip optional _idx{N} suffix
    base_label = re.sub(r"_idx\d+$", "", label)

    parts = base_label.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return label, None, None


def scan_directory(directory):
    """Scan *directory* recursively for ``*_report.log`` files.

    Returns a list of result dicts sorted by label.
    """
    results = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith("_report.log"):
                filepath = os.path.join(root, fname)
                r = parse_report(filepath)
                if r is not None:
                    base, func, basis = parse_label_for_combinatorial(
                        r["label"]
                    )
                    r["base_label"] = base
                    r["functional"] = func
                    r["basis"] = basis
                    results.append(r)

    results.sort(key=lambda x: (x.get("base_label", ""), x.get("functional") or "", x.get("basis") or ""))
    return results


CSV_FIELDS = [
    "label",
    "base_label",
    "functional",
    "basis",
    "converged",
    "n_steps",
    "e_a",
    "e_b",
    "dE_hartree",
    "dE_kcal",
    "pgrad_max",
    "pgrad_rms",
    "disp_max",
    "disp_rms",
    "step_size",
]


def write_csv(results, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def write_markdown(results, filepath):
    """Write a markdown summary table."""
    lines = [
        "| Label | Functional | Basis | Converged | Steps | E_A (Ha) | E_B (Ha) | ΔE (kcal/mol) |",
        "|-------|------------|-------|-----------|-------|----------|----------|---------------|",
    ]
    for r in results:
        conv = "✓" if r["converged"] else "✗"
        ea = f"{r['e_a']:.6f}" if r["e_a"] is not None else "N/A"
        eb = f"{r['e_b']:.6f}" if r["e_b"] is not None else "N/A"
        dk = f"{r['dE_kcal']:.3f}" if r["dE_kcal"] is not None else "N/A"
        func = r.get("functional") or "-"
        basis = r.get("basis") or "-"
        lines.append(
            f"| {r['label']} | {func} | {basis} | {conv} | {r['n_steps']} | {ea} | {eb} | {dk} |"
        )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# MECP Benchmark Summary\n\n")
        f.write("\n".join(lines))
        f.write("\n")


@click.command()
@click.option(
    "-d",
    "--directory",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory to scan for *_report.log files (recursive).",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=str,
    help="Output CSV file path. If omitted, prints to stdout.",
)
@click.option(
    "--markdown",
    default=None,
    type=str,
    help="Optional markdown summary file path.",
)
def entry_point(directory, output, markdown):
    """Aggregate chemsmart MECP benchmark results into a summary table."""
    results = scan_directory(directory)

    if not results:
        click.echo(f"No MECP report files found in {directory}")
        return

    click.echo(f"Found {len(results)} MECP report(s):\n")

    if output:
        write_csv(results, output)
        click.echo(f"CSV written to: {output}")
    else:
        # Print to stdout as CSV
        writer = csv.DictWriter(
            click.get_text_stream("stdout"),
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    if markdown:
        write_markdown(results, markdown)
        click.echo(f"Markdown written to: {markdown}")

    # Always print a quick summary
    n_conv = sum(1 for r in results if r["converged"])
    click.echo(f"\nSummary: {n_conv}/{len(results)} converged")


if __name__ == "__main__":
    entry_point()
