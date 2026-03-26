"""Runtime capability helpers."""

from __future__ import annotations

import numpy as np
from dolfinx import default_scalar_type


def is_complex_runtime() -> bool:
    """Return whether DOLFINx/PETSc was built with a complex scalar type."""
    return np.issubdtype(np.dtype(default_scalar_type), np.complexfloating)


def require_complex_runtime(preset_name: str) -> None:
    """Raise a clear error if a preset requires a complex-valued runtime."""
    if is_complex_runtime():
        return
    raise RuntimeError(
        f"Preset '{preset_name}' requires a complex-valued DOLFINx/PETSc build. "
        "Set PLM_CONDA_ENV to a complex-capable environment before running it."
    )
