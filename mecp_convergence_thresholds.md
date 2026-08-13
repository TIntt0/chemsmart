# MECP Optimization Convergence Thresholds

| Software / Source | ΔE (Hartree) | Max Force (Hartree/Bohr) | RMS Force (Hartree/Bohr) | Max Displ. (Bohr) | RMS Displ. (Bohr) |
|---|---|---|---|---|---|
| Harvey's original MECP code (de facto standard)¹ | 1 × 10⁻⁵ | 7 × 10⁻⁴ | 5 × 10⁻⁴ | 4 × 10⁻³ | 2.5 × 10⁻³ |
| EasyMECP (wrapper around Harvey's code)² | 1 × 10⁻⁵ | 7 × 10⁻⁴ | 5 × 10⁻⁴ | 4 × 10⁻³ | 2.5 × 10⁻³ |
| Qbics³ | 1 × 10⁻⁵ | 1 × 10⁻³ | — | 1 × 10⁻³ | — |
| Schrödinger Jaguar (`MECP` class)⁴ | 5 × 10⁻⁵ | 5 × 10⁻⁴ | — | — | — |
| Q-Chem (Gaussian-style opt thresholds)⁵ | 1 × 10⁻⁶ | 4.5 × 10⁻⁴ | 3 × 10⁻⁴ | 1.8 × 10⁻³ | 1.2 × 10⁻³ |
| SCM/AMS (ADF convergence)⁶ | — | — | — | — | — *(uses ADF default geometry-optimization convergence)* |
| This repository, `standard` preset⁷ | 5 × 10⁻⁵ | 7 × 10⁻⁴ | 5 × 10⁻⁴ | 4 × 10⁻³ | 2.5 × 10⁻³ |
| This repository, `tight` preset⁷ | 1 × 10⁻⁵ | 3 × 10⁻⁴ | 1 × 10⁻⁴ | 2 × 10⁻³ | 1 × 10⁻³ |

## Notes

- The Q-Chem values correspond to its standard geometry-optimization convergence criteria (applied within the MECP driver), not to separately published MECP-specific tolerances.
- SCM/AMS reuses Harvey's MECP code but applies ADF's default geometry-optimization convergence settings; no single published numerical value is therefore listed.¹ ⁶
- The `standard` preset of this repository adopts exactly the same force and displacement tolerances as Harvey's defaults; only the ΔE tolerance is slightly relaxed (5 × 10⁻⁵ vs. 1 × 10⁻⁵), while the `tight` preset is stricter than Harvey's defaults on all five criteria.⁷

## References

1. Harvey, J. N.; Aschi, M.; Schwarz, H.; Koch, W. The Singlet and Triplet States of Phenyl Cation. A Hybrid Approach for Locating Minimum Energy Crossing Points between Non-Interacting Potential Energy Surfaces. *Theor. Chem. Acc.* **1998**, *99*, 95–99. DOI: [10.1007/s002140050309](https://doi.org/10.1007/s002140050309).

2. Garcia-Granda, J. M. EasyMECP: A Self-Contained Python Wrapper for Harvey's MECP Code. GitHub Repository, 2019. https://github.com/jaimergp/easymecp (accessed Aug 13, 2026).

3. Qbics Documentation: `mecp` Keyword. https://qbics.info/doc/keywords/mecp.html (accessed Aug 13, 2026).

4. Schrödinger Release 2016-3: Schrödinger Python API — `schrodinger.application.matsci.mecp_mod.MECP` Class; Schrödinger, LLC: New York, NY, 2016. https://content.schrodinger.com/Docs/r2016-3/python_api/api/schrodinger.application.matsci.mecp_mod.MECP-class.html (accessed Aug 13, 2026).

5. Shao, Y.; et al. Q-Chem 6.4 User Manual, §9.8.4: Nonadiabatic Couplings and Optimization of Minimum-Energy Crossing Points — Job Control and Examples; Q-Chem, Inc.: Pleasanton, CA, 2024. https://manual.q-chem.com/6.4/sfdft_MECP1.html (accessed Aug 13, 2026).

6. Software for Chemistry & Materials (SCM). AMS Documentation: Minimum Energy Crossing Point. https://www.scm.com/doc.trunk/GUI/MECP.html (accessed Aug 13, 2026).

7. This repository: `chemsmart/jobs/gaussian/settings.py`, `CONVERGENCE_PRESETS` (lines 1143–1159), `standard` and `tight` presets.
