"""Tests for plm_data.core.boundary_conditions."""

import pytest
import ufl
from dolfinx import fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.config import BCConfig, DomainConfig
from plm_data.core.mesh import create_domain


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
def trial_test(V):
    """Return (u, v) trial and test functions."""
    return ufl.TrialFunction(V), ufl.TestFunction(V)


# =========================================================================
# apply_dirichlet_bcs
# =========================================================================


class TestApplyDirichletBCs:
    def test_constant_value(self, V, domain_geom):
        """Constant Dirichlet BC (value=0.0) produces a DirichletBC on the boundary."""
        bc_configs = {
            "x-": BCConfig(type="dirichlet", value=0.0),
        }
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_param_reference(self, V, domain_geom):
        """Dirichlet BC with value='param:U' resolves to the parameter value."""
        bc_configs = {
            "x+": BCConfig(type="dirichlet", value="param:U"),
        }
        parameters = {"U": 5.0}
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters)
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_spatial_field(self, V, domain_geom):
        """Dirichlet BC with a spatial field dict interpolates onto the boundary."""
        bc_configs = {
            "y+": BCConfig(
                type="dirichlet",
                value={"type": "sine_product", "params": {"ky": 1, "amplitude": 1.0}},
            ),
        }
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})
        assert len(bcs) == 1
        assert isinstance(bcs[0], fem.DirichletBC)

    def test_invalid_boundary_name(self, V, domain_geom):
        """An unknown boundary name raises ValueError."""
        bc_configs = {
            "nonexistent": BCConfig(type="dirichlet", value=0.0),
        }
        with pytest.raises(ValueError, match="Boundary 'nonexistent' not found"):
            apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})

    def test_skips_neumann(self, V, domain_geom):
        """Only Dirichlet entries are returned; Neumann entries are skipped."""
        bc_configs = {
            "x-": BCConfig(type="dirichlet", value=0.0),
            "x+": BCConfig(type="neumann", value=1.0),
            "y-": BCConfig(type="dirichlet", value=0.0),
            "y+": BCConfig(type="neumann", value=0.0),
        }
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})
        assert len(bcs) == 2
        for bc in bcs:
            assert isinstance(bc, fem.DirichletBC)

    def test_empty_when_no_dirichlet(self, V, domain_geom):
        """Returns empty list when no Dirichlet BCs are present."""
        bc_configs = {
            "x-": BCConfig(type="neumann", value=0.0),
            "x+": BCConfig(type="neumann", value=1.0),
        }
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})
        assert bcs == []

    def test_multiple_boundaries(self, V, domain_geom):
        """Dirichlet BCs on multiple boundaries each produce a separate DirichletBC."""
        bc_configs = {
            "x-": BCConfig(type="dirichlet", value=0.0),
            "x+": BCConfig(type="dirichlet", value=1.0),
            "y-": BCConfig(type="dirichlet", value=2.0),
            "y+": BCConfig(type="dirichlet", value=3.0),
        }
        bcs = apply_dirichlet_bcs(V, domain_geom, bc_configs, parameters={})
        assert len(bcs) == 4


# =========================================================================
# build_natural_bc_forms
# =========================================================================


class TestBuildNaturalBCForms:
    def test_neumann_nonzero(self, domain_geom, trial_test):
        """Neumann BC with non-zero value produces (None, L_bc) with L_bc not None."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="neumann", value=5.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is None
        assert L_bc is not None

    def test_neumann_zero_skipped(self, domain_geom, trial_test):
        """Neumann BC with zero value is skipped entirely: (None, None)."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="neumann", value=0.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is None
        assert L_bc is None

    def test_robin_nonzero(self, domain_geom, trial_test):
        """Robin BC with alpha != 0 and value != 0 produces both a_bc and L_bc."""
        u, v = trial_test
        bc_configs = {
            "y+": BCConfig(type="robin", value=2.0, alpha=3.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is not None
        assert L_bc is not None

    def test_robin_alpha_zero(self, domain_geom, trial_test):
        """Robin BC with alpha=0 produces a_bc=None, L_bc with the g term."""
        u, v = trial_test
        bc_configs = {
            "y-": BCConfig(type="robin", value=1.0, alpha=0.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is None
        assert L_bc is not None

    def test_mixed_neumann_robin(self, domain_geom, trial_test):
        """Mixed Neumann + Robin BCs both contribute to the returned forms."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="neumann", value=1.0),
            "x+": BCConfig(type="robin", value=2.0, alpha=1.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        # Robin contributes to a_bc, both contribute to L_bc
        assert a_bc is not None
        assert L_bc is not None

    def test_all_zero_neumann(self, domain_geom, trial_test):
        """All-zero Neumann BCs produce (None, None)."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="neumann", value=0.0),
            "x+": BCConfig(type="neumann", value=0.0),
            "y-": BCConfig(type="neumann", value=0.0),
            "y+": BCConfig(type="neumann", value=0.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is None
        assert L_bc is None

    def test_invalid_boundary_name(self, domain_geom, trial_test):
        """An unknown boundary name raises ValueError."""
        u, v = trial_test
        bc_configs = {
            "bogus": BCConfig(type="neumann", value=1.0),
        }
        with pytest.raises(ValueError, match="Boundary 'bogus' not found"):
            build_natural_bc_forms(u, v, domain_geom, bc_configs, parameters={})

    def test_all_dirichlet_returns_none(self, domain_geom, trial_test):
        """When all BCs are Dirichlet, natural BC forms are both None."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="dirichlet", value=0.0),
            "x+": BCConfig(type="dirichlet", value=0.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is None
        assert L_bc is None

    def test_robin_with_param_ref(self, domain_geom, trial_test):
        """Robin BC resolves param references for both alpha and value."""
        u, v = trial_test
        bc_configs = {
            "x-": BCConfig(type="robin", value="param:g", alpha="param:a"),
        }
        parameters = {"g": 2.0, "a": 0.5}
        a_bc, L_bc = build_natural_bc_forms(u, v, domain_geom, bc_configs, parameters)
        assert a_bc is not None
        assert L_bc is not None

    def test_neumann_with_param_ref(self, domain_geom, trial_test):
        """Neumann BC resolves param reference for value."""
        u, v = trial_test
        bc_configs = {
            "y+": BCConfig(type="neumann", value="param:flux"),
        }
        parameters = {"flux": 3.0}
        a_bc, L_bc = build_natural_bc_forms(u, v, domain_geom, bc_configs, parameters)
        assert a_bc is None
        assert L_bc is not None

    def test_robin_zero_value_nonzero_alpha(self, domain_geom, trial_test):
        """Robin BC with g=0 and alpha!=0 produces a_bc only (no L_bc)."""
        u, v = trial_test
        bc_configs = {
            "x+": BCConfig(type="robin", value=0.0, alpha=2.0),
        }
        a_bc, L_bc = build_natural_bc_forms(
            u, v, domain_geom, bc_configs, parameters={}
        )
        assert a_bc is not None
        assert L_bc is None

    def test_empty_bc_configs(self, domain_geom, trial_test):
        """Empty bc_configs dict returns (None, None)."""
        u, v = trial_test
        a_bc, L_bc = build_natural_bc_forms(u, v, domain_geom, {}, parameters={})
        assert a_bc is None
        assert L_bc is None
