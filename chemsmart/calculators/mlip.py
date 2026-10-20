"""ASE adapter for machine-learned interatomic potentials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MLIPCalculator:
    """Build an ASE calculator backed by MACE."""

    backend: str = "mace"
    pretrained: str | None = None
    checkpoint: str | Path | None = None
    device: str = "cpu"
    default_dtype: str = "float64"
    dispersion: bool = False

    def __post_init__(self) -> None:
        if self.backend.lower() != "mace":
            raise ValueError(f"Unsupported MLIP backend: {self.backend!r}")
        if self.pretrained is not None and self.checkpoint is not None:
            raise ValueError(
                "Choose either a pretrained model or a local checkpoint, "
                "not both."
            )
        if self.checkpoint is not None:
            checkpoint = Path(self.checkpoint).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"MACE checkpoint does not exist: {checkpoint}"
                )

    def build(self) -> Any:
        """Return the configured MACE ASE calculator."""
        try:
            from mace.calculators import MACECalculator, mace_mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "MACE is required for MLIP calculations. Install it with "
                "`pip install mace-torch`."
            ) from exc

        if self.checkpoint is not None:
            return MACECalculator(
                model_paths=str(Path(self.checkpoint).expanduser().resolve()),
                device=self.device,
                default_dtype=self.default_dtype,
            )
        return mace_mp(
            model=self.pretrained or "small",
            device=self.device,
            default_dtype=self.default_dtype,
            dispersion=self.dispersion,
        )
