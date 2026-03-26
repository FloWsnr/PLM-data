"""Boundary condition helpers for scalar and vector fields."""

import ufl
from dolfinx import fem

from plm_data.core.config import BoundaryConditionConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_vector_interpolator,
    build_ufl_field,
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
    scalar_expression_to_config,
)


def _validate_boundary_name(name: str, domain_geom: DomainGeometry) -> None:
    if name not in domain_geom.boundary_names:
        raise ValueError(
            f"Boundary '{name}' not found in domain. "
            f"Available boundaries: {list(domain_geom.boundary_names.keys())}"
        )


def _locate_boundary_dofs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    name: str,
):
    """Locate DOFs on a named boundary for the given space."""
    msh = domain_geom.mesh
    tdim = msh.topology.dim
    fdim = tdim - 1
    tag = domain_geom.boundary_names[name]
    facets = domain_geom.facet_tags.find(tag)
    return fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)


def apply_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Create DirichletBC objects for all scalar Dirichlet boundaries."""
    msh = domain_geom.mesh
    bcs = []

    for name, bc in bc_configs.items():
        if bc.type != "dirichlet":
            continue

        _validate_boundary_name(name, domain_geom)
        if bc.value.is_componentwise:
            raise ValueError("Scalar Dirichlet BCs cannot use component-wise values")

        dofs = _locate_boundary_dofs(V, domain_geom, name)
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


def apply_vector_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Create DirichletBC objects for all vector Dirichlet boundaries."""
    gdim = domain_geom.mesh.geometry.dim
    bcs = []

    for name, bc in bc_configs.items():
        if bc.type != "dirichlet":
            continue

        _validate_boundary_name(name, domain_geom)
        dofs = _locate_boundary_dofs(V, domain_geom, name)
        interp = build_vector_interpolator(bc.value, gdim, parameters)
        if interp is None:
            raise ValueError(f"Dirichlet BC on '{name}' cannot use custom values")

        bc_func = fem.Function(V)
        bc_func.interpolate(interp)
        bcs.append(fem.dirichletbc(value=bc_func, dofs=dofs))

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
        if field_config["type"] == "custom":
            raise ValueError(f"Boundary '{name}' cannot use custom scalar values")

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


def build_vector_natural_bc_forms(
    v: ufl.Argument,
    domain_geom: DomainGeometry,
    bc_configs: dict[str, BoundaryConditionConfig],
    parameters: dict[str, float],
) -> ufl.Form | None:
    """Build weak-form contributions from vector Neumann boundary conditions."""
    msh = domain_geom.mesh
    gdim = msh.geometry.dim
    L_bc = None

    for name, bc in bc_configs.items():
        if bc.type == "dirichlet":
            continue
        if bc.type == "robin":
            raise ValueError("Vector natural BCs do not support robin conditions")
        if bc.type != "neumann":
            continue

        _validate_boundary_name(name, domain_geom)
        components = component_expressions(bc.value, gdim)
        traction_terms = []
        has_nonzero = False

        for label in component_labels_for_dim(gdim):
            component_config = scalar_expression_to_config(components[label])
            if component_config["type"] == "custom":
                raise ValueError(
                    f"Boundary '{name}' component '{label}' cannot use custom values"
                )
            if component_config["type"] in ("none", "zero"):
                traction_terms.append(ufl.as_ufl(0.0))
                continue
            if component_config["type"] == "constant":
                value = resolve_param_ref(
                    component_config["params"]["value"], parameters
                )
                if value == 0.0:
                    traction_terms.append(ufl.as_ufl(0.0))
                    continue

            traction_terms.append(
                build_ufl_field(
                    msh,
                    component_config,
                    parameters,
                )
            )
            has_nonzero = True

        if not has_nonzero:
            continue

        term = ufl.inner(ufl.as_vector(traction_terms), v) * domain_geom.ds(
            domain_geom.boundary_names[name]
        )
        L_bc = term if L_bc is None else L_bc + term

    return L_bc
