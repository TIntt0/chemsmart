# MECP HPC regression outputs

These trimmed excerpts of real ORCA output files were supplied from
`C:/Users/tintt/year1/mecp/orca_mecp_test/orca_mecp`.
Each output includes the echoed input and ORCA version. Tests parse these
recorded calculations without launching ORCA. Their assertions describe the
saved runs, not a requirement that every new calculation reproduce them.

Retained lines are unchanged and remain in their original order. Comments
mark omitted sections. The excerpts retain the echoed input, ORCA version,
atom counts, convergence and termination markers, energy and gap history,
final stationary-point and subsequent coordinate tables, and both complete
frequency tables in their original order. Intermediate SCF details,
optimization geometries, and normal-mode vectors are omitted.

The complete original files remain in the source directory above. These
excerpts are parser/report fixtures, not complete calculation archives or
inputs for vibrational-mode visualization. Parsed energies, gap history,
final geometry, frequencies, and completion assessments were compared with
the full outputs when trimming and remain unchanged.

| Fixture | Original relative path | Expected assessment |
| --- | --- | --- |
| `ethylene_twisted_mecp_numfreq.out` | `testnumfreq200/ethylene_twisted_mecp_max200.out` | Converged; PES2 imaginary frequency -969.14 cm^-1; WARNING |
| `co_pathway_mecp_numfreq.out` | `testbrokenandnumfreq/orcacase3/mecp_co_pathway_mecp.out` | Converged; PES2 imaginary frequency -16.03 cm^-1; gap -0.000139537 Hartree; WARNING |
| `ooh_pathway_mecp_numfreq.out` | `orcacase4/mecp_ooh_pathway_mecp.out` | Converged; no imaginary frequencies; gap 0.000075356 Hartree; PASSED |

All three use brokenSym 1,1 and SurfCrossNumFreq. The CO and OOH cases use
M062X/maug-cc-pV(D+d)Z with SMD(acetonitrile), charge 0 and multiplicities 3/1.
Report expectations use the current default absolute energy-gap tolerance
of 0.0001 Hartree and imaginary-frequency cutoff of -1 cm^-1.
