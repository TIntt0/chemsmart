"""
ORCA Minimum Energy Cross Point (MECP) job implementation.

This module contains the ORCAMECPJob class for running minimum energy
crossing-point optimizations using ORCA's native SurfCrossOpt feature.
"""

import logging
from typing import Type

from chemsmart.jobs.orca.job import ORCAJob
from chemsmart.jobs.orca.settings import ORCAMECPJobSettings
from chemsmart.utils.periodictable import PeriodicTable

logger = logging.getLogger(__name__)


class ORCAMECPJob(ORCAJob):
    """
    ORCA Minimum Energy Cross Point (MECP) job.

    Wraps ORCA's ``SurfCrossOpt`` geometry optimisation on the crossing
    seam between two spin states of the **same charge** and using the
    **same level of theory**.  The two-state multiplicities are specified
    via the ``%mecp`` input block (PES2 multiplicity) and the ``* xyz``
    charge/multiplicity line (PES1 multiplicity).

    Attributes:
        TYPE (str): Job type identifier ('orcamecp').
        molecule: Molecule object used for the MECP optimization.
        settings: ORCAMECPJobSettings configuration for the job.
        label (str): Job identifier used for file naming.
        jobrunner: Execution backend that runs the job.
        skip_completed (bool): If True, completed jobs are not rerun.
    """

    TYPE = "orcamecp"

    @classmethod
    def settings_class(cls) -> Type[ORCAMECPJobSettings]:
        return ORCAMECPJobSettings

    def __init__(self, molecule, settings, label, jobrunner=None, **kwargs):
        """
        Initialize ORCAMECPJob.

        Args:
            molecule: Molecule object for the MECP optimization.
            settings: ORCAMECPJobSettings instance.
            label: Job label for identification.
            jobrunner: Job runner instance.
            **kwargs: Additional keyword arguments.
        """
        settings = ORCAMECPJobSettings.from_settings(settings)
        settings.validate()
        super().__init__(
            molecule=molecule,
            settings=settings,
            label=label,
            jobrunner=jobrunner,
            **kwargs,
        )
        if self.settings.mode == "numfreq" and len(self.molecule) < 3:
            raise ValueError(
                "ORCA SurfCrossNumFreq requires at least 3 atoms; "
                f"the supplied structure contains {len(self.molecule)}."
            )
        self._validate_broken_symmetry()

    def _validate_broken_symmetry(self):
        """Reject broken-symmetry requests incompatible with PES2.

        ORCA's ``brokenSym NA,NB`` describes two antiferromagnetically
        coupled centres carrying ``NA`` and ``NB`` unpaired electrons.  The
        resulting determinant has multiplicity ``abs(NA - NB) + 1``.
        """
        broken_sym = self.settings.broken_sym
        if broken_sym is None:
            return

        unpaired_a, unpaired_b = broken_sym
        broken_sym_multiplicity = abs(unpaired_a - unpaired_b) + 1
        if broken_sym_multiplicity != self.settings.multiplicity_b:
            raise ValueError(
                f"brokenSym {unpaired_a},{unpaired_b} generates PES2 "
                "multiplicity "
                f"{broken_sym_multiplicity}, but --m2 is "
                f"{self.settings.multiplicity_b}."
            )

        periodic_table = PeriodicTable()
        electron_count = (
            sum(
                periodic_table.to_atomic_number(symbol)
                for symbol in self.molecule.symbols
            )
            - self.settings.charge
        )
        if (electron_count - (broken_sym_multiplicity - 1)) % 2:
            raise ValueError(
                f"brokenSym {unpaired_a},{unpaired_b} generates "
                "multiplicity "
                f"{broken_sym_multiplicity}, which is incompatible with "
                f"the molecule's {electron_count} electrons."
            )

    @property
    def results(self):
        """Return the parsed native SurfCrossOpt result, if available."""
        output = self._output()
        return None if output is None else output.mecp_result
