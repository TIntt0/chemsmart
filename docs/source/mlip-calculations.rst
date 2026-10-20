#################
 MLIP Calculator
#################

CHEMSMART provides a command-line MLIP calculator using MACE through ASE.
The first implementation supports single-point energy and force evaluation,
geometry optimization, and molecular dynamics.

Installation
============

MACE is an optional dependency::

   pip install -e ".[mlip]"

Model selection
===============

Without a model option, CHEMSMART uses the ``small`` MACE-MP pretrained
model. A different pretrained model or a local checkpoint can be selected::

   chemsmart run mlip -f structure.xyz --pretrained medium sp
   chemsmart run mlip -f structure.xyz --checkpoint model.model sp

``--pretrained`` and ``--checkpoint`` are mutually exclusive.

Single point
============

Run::

   chemsmart run mlip -f structure.xyz -o results sp

The command writes ``mlip_sp.json`` with energy in eV and atomic forces in
eV/Angstrom.

Geometry optimization
=====================

Run::

   chemsmart run mlip -f structure.xyz -o results opt --fmax 0.05

The command writes an optimized XYZ structure, ASE trajectory, optimizer log,
and JSON summary. BFGS, L-BFGS, and FIRE are available.

Molecular dynamics
==================

Run a basic NVE trajectory with Velocity Verlet::

   chemsmart run mlip -f structure.xyz -o results md \
       --temperature-k 300 --timestep-fs 0.5 --steps 1000

The command writes an ASE trajectory, CSV energy/temperature time series, and
the final XYZ structure. Use ``--seed`` for reproducible initial velocities.
