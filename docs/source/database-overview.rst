#####################
 Database Management
#####################

CHEMSMART provides a unified database subsystem for managing, querying, and reusing quantum chemistry calculation data.
The ``chemsmart run database`` command group converts heterogeneous computational output files and calculation folders
into a structured, identity-aware, and queryable database for downstream analysis, machine learning, and CHEMSMART
workflow automation.

************
 Motivation
************

Computational chemistry projects typically generate hundreds to thousands of output files spread across different
directories, programs, and levels of theory. Without a unified data model, it can be difficult to manage, compare,
query, and report these results in a consistent and reproducible way.

The CHEMSMART database module addresses this challenge by converting fragmented calculation outputs into normalized,
reusable database records. It is designed to:

-  parse computational output files into schema-driven records;
-  separate molecular identity from calculation context;
-  deduplicate chemical species and 3D conformers across calculations;
-  store everything in a portable, single-file CHEMSMART database (``.db``);
-  provide CLI tools for inspecting, querying, filtering, and exporting data;
-  enable stored records and structures to be reused directly in downstream CHEMSMART workflows.

Together, these features allow computational results to be treated not as isolated output files, but as structured,
traceable, and reusable scientific data.

************
 Data Model
************

The CHEMSMART database separates calculation data into three core entities: ``record``, ``molecule``, and ``structure``.
These entities represent *what was computed*, *what chemical species was studied*, and *which 3D geometry was used*.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   -  -  Object
      -  Represents
      -  Description

   -  -  ``record``
      -  One calculation
      -  Identified by ``record_id``. Stores the calculation context, parsed results, runtime information, and
         provenance for one calculation.

   -  -  ``molecule``
      -  One chemical species
      -  Identified by ``molecule_id``. Stores connectivity-level molecular identity, such as formula, SMILES, InChI,
         composition, and basic molecular descriptors.

   -  -  ``structure``
      -  One 3D geometry
      -  Identified by ``structure_id``. Stores geometry-level information, including coordinates, charge, multiplicity,
         and simple geometric descriptors.

This organization allows CHEMSMART to recognize when the same molecule or conformer appears in multiple calculations,
while preserving calculation-specific information needed for analysis and reproducibility.

Identifiers
===========

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   -  -  Identifier
      -  CLI option
      -  Notes

   -  -  ``record_id``
      -  ``--rid``
      -  Hash-derived identifier for a calculation record. Prefixes of at least 12 characters are accepted.

   -  -  ``record_index``
      -  ``--ri``
      -  1-based index assigned in insertion order. Useful for quick interactive inspection.

   -  -  ``molecule_id``
      -  ``--mid``
      -  InChIKey-derived identifier for a molecular identity. Prefixes of at least 16 characters are accepted.

   -  -  ``structure_id``
      -  ``--sid``
      -  Hash-derived identifier based on geometry, charge, and multiplicity. Prefixes of at least 12 characters are
         accepted.

.. note::

   Partial IDs are resolved automatically. If a prefix uniquely matches one entry, the command proceeds; otherwise an
   explicit error is raised with the matching candidates.

**********
 Workflow
**********

A typical CHEMSMART database workflow is:

#. **Assemble** output files from a project directory into a single ``.db`` file using ``chemsmart run database
   assemble`` (see :doc:`database-assemble`).
#. **Query** database entries, with or without structured filters, using ``chemsmart run database query`` (see
   :doc:`database-query`).
#. **Inspect** selected entries in detail using ``chemsmart run database inspect`` (see :doc:`database-inspect`).
#. **Export** selected content as JSON, CSV, XYZ, or extended XYZ (with per-atom forces) for downstream analysis using
   ``chemsmart run database export`` (see :doc:`database-export`).
#. **Reuse** database entries in downstream chemsmart workflows (see :doc:`database-workflow`).

*****************
 Command Summary
*****************

.. list-table::
   :header-rows: 1
   :widths: 20 80

   -  -  Subcommand
      -  Purpose
   -  -  ``assemble``
      -  Parse computational chemistry output files and build a CHEMSMART database.
   -  -  ``inspect``
      -  Print detailed information for databases, records, molecules, and structures.
   -  -  ``query``
      -  Filter records, molecules, or structures using a structured query expression.
   -  -  ``export``
      -  Export the database or selected entries to JSON, CSV, XYZ, or extended XYZ.

*************
 Usage Notes
*************

-  **Supported programs.** Gaussian and ORCA output files and xTB calculation folders are currently parsed.
   Files/folders from other programs are skipped during ``assemble``. Note that xTB results are organized as calculation
   folders rather than single output files.

-  **Idempotent assembly.** Re-running ``assemble`` on the same directory will not duplicate records. Entries with the
   same ``record_id`` are replaced, and the number of replaced duplicates is reported.

-  **Database validation.** Every database operation first verifies that the input is a valid CHEMSMART database with
   the expected schema. Foreign ``.db`` files are rejected with a clear error.

-  **Portability.** A CHEMSMART database is stored as a single portable ``.db`` file. It can be copied between machines
   and shared alongside the source outputs for reproducibility, since each record carries provenance metadata.

***********************
 Important Assumptions
***********************

-  Molecular identity depends on the connectivity information that can be inferred from the parsed output. For unusual
   bonding situations, organometallic complexes, ion pairs, or transition states, users should inspect the assigned
   molecule and structure identifiers carefully.

-  Structure identity is geometry-based and includes charge and multiplicity. The same Cartesian geometry with different
   charge or spin state is treated as a different structure.

-  Molecular descriptors such as ``is_aromatic``, ``is_multicomponent``, ``smiles``, and ``inchi`` are derived from 3D
   geometry using RDKit heuristics, not from first-principles electronic structure. Bond orders are inferred from
   interatomic distances, which may give incorrect results for unusual bonding, metal-ligand interactions, or strained
   systems. These tags should be treated as approximate.

***************************************
 Validated structure fields and repair
***************************************

New ``chemsmart run database assemble`` imports derive SMILES, InChI, ring membership, aromaticity, chirality and
fragment counts from one RDKit molecular graph, using the geometry and molecular charge. Failed conversions retain the
geometry and calculation data, store null derived fields, and record the error in ``structure_conversions``. Existing
nonempty fields are not automatically replaced during assembly.

To inspect an existing database, first run a preview:

.. code::

   chemsmart run database repair-structure -f molecules.db \
       --scope all --limit 100 --report preview.jsonl

Inspect the JSONL report, then check the entire database:

.. code::

   chemsmart run database repair-structure -f molecules.db \
       --scope all --report full-preview.jsonl

To write repairs, use a new report filename and explicitly add ``--apply``:

.. code::

   chemsmart run database repair-structure -f molecules.db \
       --scope all --report applied.jsonl --apply

The default is a read-only preview. ``--database`` is an alias for ``-f``. Reports must use a new filename. ``--scope
invalid`` selects nonempty SMILES that cannot be parsed; ``--scope missing`` selects absent or empty SMILES. ``--scope
all`` is recommended because parseable SMILES may still encode the wrong structure. ``--limit`` counts molecular
species; all geometries of each selected species are inspected.

The repair rebuilds SMILES, InChI, ``chiral_centers``, ``is_chiral``, ``is_ring``, ``is_aromatic``,
``is_multicomponent`` and ``num_components``. All geometries belonging to a species must agree before its shared fields
are updated. Failures and conflicts are reported without changing those fields. Chiral-center indices refer to the input
atom order; disagreements in these indices also require review. Geometry, energies, forces, molecular and structural
IDs, and record links remain intact. Existing IDs are not recomputed or merged by this command.

Each applied species update and its audit history are committed together. The ``structure_conversions`` table retains
old and new values, status, timestamp and RDKit/method version. If a long run is interrupted, already committed species
remain repaired; a fresh run can check them again.

Bond inference is not guaranteed for unusual valence states, radicals, metals or transition states. Conversion failures
must be reviewed rather than treated as failed electronic-structure calculations.
