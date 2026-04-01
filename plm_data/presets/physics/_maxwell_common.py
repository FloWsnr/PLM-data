"""Shared helpers for Maxwell-family presets."""

import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import apply_vector_dirichlet_bcs
from plm_data.core.config import BoundaryFieldConfig, FieldExpressionConfig
from plm_data.core.mesh import DomainGeometry
from plm_data.core.spatial_fields import (
    build_vector_interpolator,
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
    scalar_expression_to_config,
)


def _as_3d_vector(value, gdim: int):
    """Embed a 2D tangential quantity in 3D for cross products."""
    if gdim == 2:
        return ufl.as_vector((value[0], value[1], 0))
    return value


def tangential_inner(u, v, n, gdim: int):
    """Return the tangential inner product on a boundary."""
    u_3d = _as_3d_vector(u, gdim)
    v_3d = _as_3d_vector(v, gdim)
    n_3d = _as_3d_vector(n, gdim)
    return ufl.inner(ufl.cross(u_3d, n_3d), ufl.cross(v_3d, n_3d))


def apply_maxwell_dirichlet_bcs(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Apply strong PEC-style Maxwell boundary conditions."""
    _validate_boundary_types(
        boundary_field,
        allowed={"dirichlet", "absorbing", "periodic"},
    )
    return apply_vector_dirichlet_bcs(V, domain_geom, boundary_field, parameters)


def apply_split_maxwell_dirichlet_bcs(
    mixed_space: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
    parameters: dict[str, float],
) -> list[fem.DirichletBC]:
    """Apply PEC-style Dirichlet data to the real part and zero to the imaginary."""
    _validate_boundary_types(
        boundary_field,
        allowed={"dirichlet", "absorbing", "periodic"},
    )

    gdim = domain_geom.mesh.geometry.dim
    fdim = domain_geom.mesh.topology.dim - 1
    real_subspace = mixed_space.sub(0)
    real_space, _ = real_subspace.collapse()
    imag_subspace = mixed_space.sub(1)
    imag_space, _ = imag_subspace.collapse()
    bcs: list[fem.DirichletBC] = []

    for name, entries in boundary_field.sides.items():
        for bc in entries:
            if bc.type != "dirichlet":
                continue
            if bc.value is None:
                raise ValueError(f"Dirichlet BC on '{name}' requires a value")

            facets = domain_geom.facet_tags.find(domain_geom.boundary_names[name])
            real_dofs = fem.locate_dofs_topological(
                (real_subspace, real_space), fdim, facets
            )
            imag_dofs = fem.locate_dofs_topological(
                (imag_subspace, imag_space), fdim, facets
            )

            interp = build_vector_interpolator(bc.value, gdim, parameters)
            if interp is None:
                raise ValueError(f"Dirichlet BC on '{name}' cannot use custom values")

            real_bc = fem.Function(real_space)
            real_bc.interpolate(interp)
            bcs.append(fem.dirichletbc(real_bc, real_dofs, real_subspace))

            imag_bc = fem.Function(imag_space)
            imag_bc.x.array[:] = 0.0
            bcs.append(fem.dirichletbc(imag_bc, imag_dofs, imag_subspace))

    return bcs


def build_absorbing_boundary_form(
    u,
    v,
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
    coefficient,
    parameters: dict[str, float],
):
    """Build homogeneous absorbing boundary contributions."""
    _validate_boundary_types(
        boundary_field,
        allowed={"dirichlet", "absorbing", "periodic"},
    )

    gdim = domain_geom.mesh.geometry.dim
    n = ufl.FacetNormal(domain_geom.mesh)
    form = None
    for name, entries in boundary_field.sides.items():
        for bc in entries:
            if bc.type != "absorbing":
                continue
            assert bc.value is not None
            _require_zero_absorbing_value(name, bc.value, gdim, parameters)
            term = (
                coefficient
                * tangential_inner(u, v, n, gdim)
                * domain_geom.ds(domain_geom.boundary_names[name])
            )
            form = term if form is None else form + term
    return form


def _validate_boundary_types(
    boundary_field: BoundaryFieldConfig,
    allowed: set[str],
) -> None:
    for name, entries in boundary_field.sides.items():
        for bc in entries:
            if bc.type not in allowed:
                allowed_str = ", ".join(sorted(allowed))
                raise ValueError(
                    f"Maxwell boundary '{name}' uses unsupported type '{bc.type}'. "
                    f"Allowed types: {allowed_str}."
                )


def _require_zero_absorbing_value(
    name: str,
    expr: FieldExpressionConfig,
    gdim: int,
    parameters: dict[str, float],
) -> None:
    """Ensure v1 absorbing boundaries are homogeneous."""
    components = component_expressions(expr, gdim)
    for label in component_labels_for_dim(gdim):
        component = scalar_expression_to_config(components[label])
        expr_type = component["type"]
        if expr_type in {"none", "zero"}:
            continue
        if expr_type == "constant":
            value = resolve_param_ref(component["params"]["value"], parameters)
            if value == 0.0:
                continue
        raise ValueError(
            f"Absorbing boundary '{name}' must use a zero value in v1. "
            f"Component '{label}' was configured with '{expr_type}'."
        )
