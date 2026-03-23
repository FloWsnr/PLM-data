"""Boundary condition application from config.

Converts BCConfig entries into DOLFINx DirichletBC objects and/or
weak-form contributions (Neumann, Robin), using DomainGeometry for
boundary identification and spatial_fields for value resolution.

Supported BC types:
  - dirichlet: strong constraint via fem.dirichletbc()
  - neumann:   ∂u/∂n = g  →  adds g*v*ds to L
  - robin:     ∂u/∂n + α*u = g  →  adds α*u*v*ds to a, g*v*ds to L
"""

import ufl
from dolfinx import fem

from plm_data.core.config import BCConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_ufl_field,
    normalize_field_config,
    resolve_param_ref,
)


def _validate_boundary_name(name: str, domain_geom: DomainGeometry):
    if name not in domain_geom.boundary_names:
        raise ValueError(
            f"Boundary '{name}' not found in domain. "
            f"Available boundaries: {list(domain_geom.boundary_names.keys())}"
        )


def apply_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BCConfig],
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Create DirichletBC objects for all Dirichlet boundaries.

    Args:
        V: The function space.
        domain_geom: Domain geometry with tagged boundaries.
        bc_configs: Mapping from boundary name to BCConfig.
        parameters: PDE parameters for resolving 'param:name' refs.

    Returns:
        List of DirichletBC objects.
    """
    msh = domain_geom.mesh
    tdim = msh.topology.dim
    fdim = tdim - 1
    bcs = []

    for name, bc in bc_configs.items():
        if bc.type != "dirichlet":
            continue

        _validate_boundary_name(name, domain_geom)

        tag = domain_geom.boundary_names[name]
        facets = domain_geom.facet_tags.find(tag)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)

        field_config = normalize_field_config(bc.value)

        if field_config["type"] == "constant":
            value = resolve_param_ref(field_config["params"]["value"], parameters)
            bc_obj = fem.dirichletbc(
                value=fem.Constant(msh, float(value)), dofs=dofs, V=V
            )
        else:
            interp = build_interpolator(field_config, parameters)
            assert interp is not None, (
                f"No interpolator for field type '{field_config['type']}'"
            )
            bc_func = fem.Function(V)
            bc_func.interpolate(interp)  # type: ignore[arg-type]
            bc_obj = fem.dirichletbc(value=bc_func, dofs=dofs)  # type: ignore[arg-type]

        bcs.append(bc_obj)

    return bcs


def build_natural_bc_forms(
    u: ufl.Argument,
    v: ufl.Argument,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BCConfig],
    parameters: dict[str, float],
) -> tuple[ufl.Form | None, ufl.Form | None]:
    """Build weak-form contributions from Neumann and Robin BCs.

    Neumann (∂u/∂n = g):  adds g*v*ds(tag) to L
    Robin (∂u/∂n + α*u = g):  adds α*u*v*ds(tag) to a, g*v*ds(tag) to L

    Args:
        u: The trial function.
        v: The test function.
        domain_geom: Domain geometry with tagged boundaries and ds measure.
        bc_configs: Mapping from boundary name to BCConfig.
        parameters: PDE parameters for resolving 'param:name' refs.

    Returns:
        (a_bc, L_bc) tuple. Either may be None if no contributions exist.
        a_bc should be added to the bilinear form, L_bc to the linear form.
    """
    msh = domain_geom.mesh
    a_bc = None
    L_bc = None

    for name, bc in bc_configs.items():
        if bc.type not in ("neumann", "robin"):
            continue

        _validate_boundary_name(name, domain_geom)
        tag = domain_geom.boundary_names[name]

        # --- L contribution: g * v * ds(tag) ---
        field_config = normalize_field_config(bc.value)

        # Skip zero values (no assembly needed)
        skip_L = False
        if field_config["type"] in ("none", "zero"):
            skip_L = True
        elif field_config["type"] == "constant":
            val = resolve_param_ref(field_config["params"]["value"], parameters)
            if val == 0.0:
                skip_L = True

        if not skip_L:
            g = build_ufl_field(msh, field_config, parameters)
            term = ufl.inner(g, v) * domain_geom.ds(tag)
            L_bc = term if L_bc is None else L_bc + term  # type: ignore[reportOperatorIssue]

        # --- a contribution (Robin only): α * u * v * ds(tag) ---
        if bc.type == "robin":
            alpha = resolve_param_ref(bc.alpha, parameters)
            if alpha != 0.0:
                term = alpha * ufl.inner(u, v) * domain_geom.ds(tag)  # type: ignore[reportOperatorIssue]
                a_bc = term if a_bc is None else a_bc + term  # type: ignore[reportOperatorIssue]

    return a_bc, L_bc
