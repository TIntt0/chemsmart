"""
ORCA Minimum Energy Cross Point (MECP) job implementation.

This module contains the ORCAMECPJob class for running minimum energy
crossing-point optimizations using ORCA's native SurfCrossOpt feature.
"""

import logging
from typing import Type

from chemsmart.jobs.orca.job import ORCAJob
from chemsmart.jobs.orca.settings import ORCAMECPJobSettings

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

    @property
    def results(self):
        """Return the parsed native SurfCrossOpt result, if available."""
        output = self._output()
        return None if output is None else output.mecp_result
