"""Initial condition generators for DOLFINx functions.

Delegates to the shared spatial field type system for spatial IC types
(constant, gaussian_bump, sine_product, step). Handles random_perturbation
directly (DOF-based, not a spatial field).
"""

import numpy as np
from dolfinx import fem

from plm_data.core.config import ICConfig
from plm_data.core.spatial_fields import build_interpolator


def apply_ic(
    func: fem.Function,
    ic_config: ICConfig,
    parameters: dict[str, float],
    seed: int | None = None,
) -> None:
    """Apply an initial condition to a DOLFINx Function in-place.

    Spatial IC types (constant, gaussian_bump, sine_product, step) are
    delegated to the shared spatial_fields system. random_perturbation
    is handled directly as it operates on DOF arrays, not spatial coords.

    Args:
        func: The Function to initialize.
        ic_config: IC configuration with type and params.
        parameters: PDE parameters for resolving 'param:name' refs.
        seed: Random seed for reproducible ICs.
    """
    ic_type = ic_config.type

    if ic_type == "custom":
        # Preset is responsible for generating its own IC.
        return

    if ic_type == "random_perturbation":
        # DOF-based — not a spatial field, needs RNG.
        rng = np.random.default_rng(seed)
        p = ic_config.params
        mean = p["mean"]
        std = p["std"]
        values = rng.normal(mean, std, size=func.x.array.shape)
        func.x.array[:] = values
        return

    # All other types: delegate to the shared spatial field system.
    field_config = {"type": ic_type, "params": ic_config.params}
    interpolator = build_interpolator(field_config, parameters)
    if interpolator is not None:
        func.interpolate(interpolator)


def apply_vector_ic(
    vector_func: fem.Function,
    component_funcs: list[fem.Function],
    component_dofs: list[np.ndarray],
    ic_configs: dict[str, ICConfig],
    component_names: list[str],
    parameters: dict[str, float],
    seed: int | None = None,
) -> None:
    """Apply per-component ICs to a vector function.

    Applies scalar ICs to each named component, assembles the results
    into the vector function. Used by vector PDEs (Navier-Stokes,
    elasticity, etc.) where ICs are specified per scalar component.

    Args:
        vector_func: The vector Function to populate (e.g., DG Lagrange vector).
        component_funcs: Scalar Functions for each component (collapsed sub-spaces).
        component_dofs: DOF index arrays mapping each component into vector_func.
        ic_configs: Per-component IC configs keyed by component name.
        component_names: Names matching ic_configs keys (e.g., ["velocity_x", "velocity_y"]).
        parameters: PDE parameters for resolving 'param:name' refs.
        seed: Random seed for reproducible ICs.
    """
    for name, comp_func in zip(component_names, component_funcs):
        if name in ic_configs:
            apply_ic(comp_func, ic_configs[name], parameters, seed=seed)

    # Assemble scalar components into the vector function
    for comp_func, dofs in zip(component_funcs, component_dofs):
        vector_func.x.array[dofs] = comp_func.x.array
