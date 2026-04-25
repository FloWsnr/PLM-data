"""Tests for shared PDE helper builders."""

from plm_data.pdes.fisher_kpp.helpers import build_scalar_reaction_diffusion_spec
from plm_data.pdes.metadata import PDEParameter


def test_build_scalar_reaction_diffusion_spec_creates_standard_spec():
    spec = build_scalar_reaction_diffusion_spec(
        name="unit_reaction_diffusion",
        description="test reaction diffusion",
        reaction_equation="u * (1 - u)",
        parameters=[PDEParameter("r", "reaction rate")],
    )

    assert spec.name == "unit_reaction_diffusion"
    assert spec.category == "biology"
    assert [parameter.name for parameter in spec.parameters] == ["D", "r"]
    assert set(spec.inputs) == {"u"}
    assert set(spec.boundary_fields) == {"u"}
    assert set(spec.states) == {"u"}
    assert set(spec.outputs) == {"u"}
    assert set(spec.coefficients) == {"velocity"}
