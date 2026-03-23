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
