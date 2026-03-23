"""Initial condition generators for DOLFINx functions."""

import numpy as np
from dolfinx import fem

from plm_data.core.config import ICConfig


def _require_param(params: dict, key: str, ic_type: str):
    """Require a parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for initial condition '{ic_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def apply_ic(
    func: fem.Function,
    ic_config: ICConfig,
    seed: int | None = None,
) -> None:
    """Apply an initial condition to a DOLFINx Function in-place.

    All parameters must be explicitly provided in ic_config.params.

    Args:
        func: The Function to initialize.
        ic_config: IC configuration with type and params.
        seed: Random seed for reproducible ICs.
    """
    rng = np.random.default_rng(seed)
    ic_type = ic_config.type
    p = ic_config.params

    if ic_type == "custom":
        # Preset is responsible for generating its own IC.
        return

    elif ic_type == "gaussian_bump":
        sigma = _require_param(p, "sigma", ic_type)
        amplitude = _require_param(p, "amplitude", ic_type)
        cx = _require_param(p, "cx", ic_type)
        cy = _require_param(p, "cy", ic_type)
        func.interpolate(
            lambda x: amplitude * np.exp(-((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / (2 * sigma**2))
        )

    elif ic_type == "random_perturbation":
        mean = _require_param(p, "mean", ic_type)
        std = _require_param(p, "std", ic_type)
        values = rng.normal(mean, std, size=func.x.array.shape)
        func.x.array[:] = values

    elif ic_type == "sine_wave":
        kx = _require_param(p, "kx", ic_type)
        ky = _require_param(p, "ky", ic_type)
        amplitude = _require_param(p, "amplitude", ic_type)
        func.interpolate(
            lambda x: amplitude * np.sin(kx * np.pi * x[0]) * np.sin(ky * np.pi * x[1])
        )

    elif ic_type == "constant":
        value = _require_param(p, "value", ic_type)
        func.x.array[:] = value

    elif ic_type == "step":
        value_left = _require_param(p, "value_left", ic_type)
        value_right = _require_param(p, "value_right", ic_type)
        x_split = _require_param(p, "x_split", ic_type)
        func.interpolate(
            lambda x: np.where(x[0] < x_split, value_left, value_right)
        )

    else:
        raise ValueError(f"Unknown initial condition type: '{ic_type}'")
