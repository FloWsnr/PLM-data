"""Source term construction from config.

Builds the f * v * dx contribution to the linear form using the shared
spatial field type system.
"""

import ufl
from dolfinx import mesh as dmesh

from plm_data.core.config import SourceTermConfig
from plm_data.core.spatial_fields import build_ufl_field


def build_source_form(
    v: ufl.Argument,
    msh: dmesh.Mesh,
    source_config: SourceTermConfig,
    parameters: dict[str, float],
) -> ufl.Form | None:
    """Build the source term contribution to the linear form.

    Args:
        v: The test function.
        msh: The mesh.
        source_config: Source term configuration with type and params.
        parameters: PDE parameters for resolving 'param:name' refs.

    Returns:
        A UFL form f * v * dx, or None for type "none" or "custom".
    """
    field_config = {"type": source_config.type, "params": source_config.params}

    if source_config.type in ("none", "custom"):
        return None

    f = build_ufl_field(msh, field_config, parameters)
    return ufl.inner(f, v) * ufl.dx


def build_vector_source_form(
    v: ufl.Argument,
    msh: dmesh.Mesh,
    component_names: list[str],
    source_configs: dict[str, SourceTermConfig],
    parameters: dict[str, float],
) -> ufl.Form | None:
    """Build a vector source term from per-component scalar configs.

    Assembles per-component scalar source terms into a vector body force
    and returns inner(f_body, v) * dx. Used by vector PDEs (Navier-Stokes,
    elasticity, etc.) where source terms are specified per scalar component.

    Args:
        v: The vector test function.
        msh: The mesh.
        component_names: Component names matching source_configs keys
            (e.g., ["velocity_x", "velocity_y"]).
        source_configs: Per-component source term configs keyed by name.
        parameters: PDE parameters for resolving 'param:name' refs.

    Returns:
        A UFL form inner(f_body, v) * dx, or None if all components are
        none/zero.
    """
    components = []
    has_nonzero = False
    for name in component_names:
        st_config = source_configs.get(name)
        if st_config and st_config.type not in ("none", "custom"):
            field_config = {"type": st_config.type, "params": st_config.params}
            f_comp = build_ufl_field(msh, field_config, parameters)
            components.append(f_comp)
            has_nonzero = True
        else:
            components.append(ufl.as_ufl(0.0))

    if not has_nonzero:
        return None

    f_body = ufl.as_vector(components)
    return ufl.inner(f_body, v) * ufl.dx
