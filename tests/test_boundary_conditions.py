"""Tests for plm_data.boundary_conditions.runtime."""

import pytest
import ufl
from dolfinx import fem

from plm_data.boundary_conditions.runtime import (
    apply_dirichlet_bcs,
    apply_vector_dirichlet_bcs,
    build_natural_bc_forms,
    build_vector_natural_bc_forms,
)
from plm_data.core.runtime_config import (
    BoundaryConditionConfig,
    DomainConfig,
    FieldExpressionConfig,
)
from plm_data.domains import create_domain
from tests.runtime_helpers import boundary_field_config


def constant(value):
    return FieldExpressionConfig(type="constant", params={"value": value})


def vector_constant(*values):
    labels = ("x", "y", "z")[: len(values)]
    return FieldExpressionConfig(
        components={
            label: constant(value) for label, value in zip(labels, values, strict=False)
        }
    )


@pytest.fixture
def domain_geom():
    domain = DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [4, 4]},
    )
    return create_domain(domain)


@pytest.fixture
def V(domain_geom):
    return fem.functionspace(domain_geom.mesh, ("Lagrange", 1))


@pytest.fixture
def V_vec(domain_geom):
    return fem.functionspace(domain_geom.mesh, ("Lagrange", 1, (2,)))


@pytest.fixture
def trial_test(V):
    return ufl.TrialFunction(V), ufl.TestFunction(V)


@pytest.fixture
def vector_test(V_vec):
    return ufl.TestFunction(V_vec)


class TestApplyDirichletBCs:
    def test_constant_value(self, V, domain_geom):
        bc_configs = {
            "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0))
        }
        bcs = apply_dirichlet_bcs(
            V, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_param_reference(self, V, domain_geom):
        bc_configs = {
            "x+": BoundaryConditionConfig(type="dirichlet", value=constant("param:U")),
        }
        bcs = apply_dirichlet_bcs(
            V, domain_geom, boundary_field_config(bc_configs), {"U": 5.0}
        )
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_spatial_field(self, V, domain_geom):
        bc_configs = {
            "y+": BoundaryConditionConfig(
                type="dirichlet",
                value=FieldExpressionConfig(
                    type="sine_waves",
                    params={
                        "background": 0.0,
                        "modes": [
                            {"amplitude": 1.0, "cycles": [0.0, 1.0], "phase": 0.0}
                        ],
                    },
                ),
            ),
        }
        bcs = apply_dirichlet_bcs(
            V, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_invalid_boundary_name(self, V, domain_geom):
        bc_configs = {
            "nonexistent": BoundaryConditionConfig(
                type="dirichlet", value=constant(0.0)
            ),
        }
        with pytest.raises(ValueError, match="Boundary 'nonexistent' not found"):
            apply_dirichlet_bcs(
                V, domain_geom, boundary_field_config(bc_configs), parameters={}
            )

    def test_skips_neumann(self, V, domain_geom):
        bc_configs = {
            "x-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(1.0)),
            "y-": BoundaryConditionConfig(type="dirichlet", value=constant(0.0)),
            "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        }
        bcs = apply_dirichlet_bcs(
            V, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert len(bcs) == 2

    def test_empty_when_no_dirichlet(self, V, domain_geom):
        bc_configs = {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
            "x+": BoundaryConditionConfig(type="neumann", value=constant(1.0)),
        }
        assert (
            apply_dirichlet_bcs(
                V, domain_geom, boundary_field_config(bc_configs), parameters={}
            )
            == []
        )


class TestApplyVectorDirichletBCs:
    def test_constant_value(self, V_vec, domain_geom):
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="dirichlet", value=vector_constant(0.0, 1.0)
            )
        }
        bcs = apply_vector_dirichlet_bcs(
            V_vec, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_spatial_field(self, V_vec, domain_geom):
        bc_configs = {
            "y+": BoundaryConditionConfig(
                type="dirichlet",
                value=FieldExpressionConfig(
                    components={
                        "x": FieldExpressionConfig(
                            type="sine_waves",
                            params={
                                "background": 0.0,
                                "modes": [
                                    {
                                        "amplitude": 1.0,
                                        "cycles": [0.0, 1.0],
                                        "phase": 0.0,
                                    }
                                ],
                            },
                        ),
                        "y": constant(0.0),
                    }
                ),
            ),
        }
        bcs = apply_vector_dirichlet_bcs(
            V_vec, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_invalid_boundary_name(self, V_vec, domain_geom):
        bc_configs = {
            "nonexistent": BoundaryConditionConfig(
                type="dirichlet", value=vector_constant(0.0, 0.0)
            ),
        }
        with pytest.raises(ValueError, match="Boundary 'nonexistent' not found"):
            apply_vector_dirichlet_bcs(
                V_vec, domain_geom, boundary_field_config(bc_configs), parameters={}
            )

    def test_empty_when_no_dirichlet(self, V_vec, domain_geom):
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="neumann", value=vector_constant(0.0, 0.0)
            ),
            "x+": BoundaryConditionConfig(
                type="neumann", value=vector_constant(1.0, 0.0)
            ),
        }
        assert (
            apply_vector_dirichlet_bcs(
                V_vec, domain_geom, boundary_field_config(bc_configs), parameters={}
            )
            == []
        )


class TestBuildNaturalBCForms:
    def test_neumann_nonzero(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(5.0)),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert a_bc is None
        assert L_bc is not None

    def test_neumann_zero_skipped(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert a_bc is None
        assert L_bc is None

    def test_robin_nonzero(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "y+": BoundaryConditionConfig(type="robin", value=constant(2.0), alpha=3.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert a_bc is not None
        assert L_bc is not None

    def test_robin_alpha_zero(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "y-": BoundaryConditionConfig(type="robin", value=constant(1.0), alpha=0.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
        )
        assert a_bc is None
        assert L_bc is not None

    def test_invalid_boundary_name(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "bogus": BoundaryConditionConfig(type="neumann", value=constant(1.0)),
        }
        with pytest.raises(ValueError, match="Boundary 'bogus' not found"):
            build_natural_bc_forms(
                u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
            )

    def test_robin_with_param_ref(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="robin",
                value=constant("param:g"),
                alpha="param:a",
            ),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u,
            v,
            domain_geom,
            boundary_field_config(bc_configs),
            {"g": 2.0, "a": 0.5},
        )
        assert a_bc is not None
        assert L_bc is not None

    def test_custom_value_raises_clear_error(self, domain_geom, trial_test):
        u, v = trial_test
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="neumann",
                value=FieldExpressionConfig(type="custom", params={}),
            ),
        }
        with pytest.raises(
            ValueError, match="Boundary 'x-' cannot use custom scalar values"
        ):
            build_natural_bc_forms(
                u, v, domain_geom, boundary_field_config(bc_configs), parameters={}
            )


class TestBuildVectorNaturalBCForms:
    def test_neumann_nonzero(self, domain_geom, vector_test):
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="neumann", value=vector_constant(5.0, 0.0)
            ),
        }
        L_bc = build_vector_natural_bc_forms(
            vector_test,
            domain_geom,
            boundary_field_config(bc_configs),
            parameters={},
        )
        assert L_bc is not None

    def test_neumann_zero_skipped(self, domain_geom, vector_test):
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="neumann", value=vector_constant(0.0, 0.0)
            ),
        }
        L_bc = build_vector_natural_bc_forms(
            vector_test,
            domain_geom,
            boundary_field_config(bc_configs),
            parameters={},
        )
        assert L_bc is None

    def test_param_reference(self, domain_geom, vector_test):
        bc_configs = {
            "y+": BoundaryConditionConfig(
                type="neumann",
                value=vector_constant("param:tx", "param:ty"),
            ),
        }
        L_bc = build_vector_natural_bc_forms(
            vector_test,
            domain_geom,
            boundary_field_config(bc_configs),
            parameters={"tx": 2.0, "ty": -1.0},
        )
        assert L_bc is not None

    def test_robin_raises(self, domain_geom, vector_test):
        bc_configs = {
            "y+": BoundaryConditionConfig(
                type="robin",
                value=vector_constant(1.0, 0.0),
                alpha=3.0,
            ),
        }
        with pytest.raises(
            ValueError, match="Shared vector Robin is intentionally unsupported"
        ):
            build_vector_natural_bc_forms(
                vector_test,
                domain_geom,
                boundary_field_config(bc_configs),
                parameters={},
            )

    def test_custom_component_raises(self, domain_geom, vector_test):
        bc_configs = {
            "x-": BoundaryConditionConfig(
                type="neumann",
                value=FieldExpressionConfig(
                    components={
                        "x": FieldExpressionConfig(type="custom", params={}),
                        "y": constant(0.0),
                    }
                ),
            ),
        }
        with pytest.raises(
            ValueError,
            match="Boundary 'x-' component 'x' cannot use custom values",
        ):
            build_vector_natural_bc_forms(
                vector_test,
                domain_geom,
                boundary_field_config(bc_configs),
                parameters={},
            )
