"""Boundary condition helpers for scalar fields."""

import ufl
from dolfinx import fem

from plm_data.core.config import BoundaryConditionConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_ufl_field,
    resolve_param_ref,
    scalar_expression_to_config,
)


def _validate_boundary_name(name: str, domain_geom: DomainGeometry) -> None:
    if name not in domain_geom.boundary_names:
        raise ValueError(
            f"Boundary '{name}' not found in domain. "
            f"Available boundaries: {list(domain_geom.boundary_names.keys())}"
        )


def apply_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Create DirichletBC objects for all scalar Dirichlet boundaries."""
    msh = domain_geom.mesh
    tdim = msh.topology.dim
    fdim = tdim - 1
    bcs = []

    for name, bc in bc_configs.items():
        if bc.type != "dirichlet":
            continue

        _validate_boundary_name(name, domain_geom)
        if bc.value.is_componentwise:
            raise ValueError("Scalar Dirichlet BCs cannot use component-wise values")

        tag = domain_geom.boundary_names[name]
        facets = domain_geom.facet_tags.find(tag)
        dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
        field_config = scalar_expression_to_config(bc.value)

        if field_config["type"] == "constant":
            value = resolve_param_ref(field_config["params"]["value"], parameters)
            bc_obj = fem.dirichletbc(
                value=fem.Constant(msh, float(value)),
                dofs=dofs,
                V=V,
            )
        else:
            interp = build_interpolator(field_config, parameters)
            if interp is None:
                raise ValueError(f"Dirichlet BC on '{name}' cannot use custom values")
            bc_func = fem.Function(V)
            bc_func.interpolate(interp)
            bc_obj = fem.dirichletbc(value=bc_func, dofs=dofs)

        bcs.append(bc_obj)

    return bcs


def build_natural_bc_forms(
    u: ufl.Argument,
    v: ufl.Argument,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> tuple[ufl.Form | None, ufl.Form | None]:
    """Build weak-form contributions from scalar Neumann and Robin BCs."""
    msh = domain_geom.mesh
    a_bc = None
    L_bc = None

    for name, bc in bc_configs.items():
        if bc.type not in ("neumann", "robin"):
            continue

        _validate_boundary_name(name, domain_geom)
        if bc.value.is_componentwise:
            raise ValueError("Scalar natural BCs cannot use component-wise values")

        tag = domain_geom.boundary_names[name]
        field_config = scalar_expression_to_config(bc.value)

        skip_L = False
        if field_config["type"] in ("none", "zero"):
            skip_L = True
        elif field_config["type"] == "constant":
            value = resolve_param_ref(field_config["params"]["value"], parameters)
            if value == 0.0:
                skip_L = True

        if not skip_L:
            g = build_ufl_field(msh, field_config, parameters)
            term = ufl.inner(g, v) * domain_geom.ds(tag)
            L_bc = term if L_bc is None else L_bc + term

        if bc.type == "robin":
            alpha = resolve_param_ref(bc.alpha, parameters)
            if alpha != 0.0:
                term = alpha * ufl.inner(u, v) * domain_geom.ds(tag)
                a_bc = term if a_bc is None else a_bc + term

    return a_bc, L_bc
