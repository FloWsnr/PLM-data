"""OpenFOAM-backed magnetohydrodynamics preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.fluids._openfoam import OpenFOAMMHDProblem
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)


_MHD_SPEC = PresetSpec(
    name="mhd",
    category="fluids",
    description=(
        "OpenFOAM `mhdFoam` formulation for incompressible, laminar "
        "magnetohydrodynamics with magnetic flux pressure correction on "
        "supported OpenFOAM meshes. Fully periodic magnetic domains are "
        "excluded because the stock solver requires a `pB` anchor."
    ),
    equations={
        "velocity": (
            "dU/dt + div(phi,U) - div(phiB,(1/(mu*rho))*B) = "
            "-grad(p) + nu*laplacian(U) + grad((1/(2*mu*rho))*|B|^2)"
        ),
        "pressure": "div(U) = 0",
        "magnetic_field": "dB/dt + div(phi,B) - div(phiB,U) = DB*laplacian(B)",
        "magnetic_constraint": (
            "pB: magnetic flux pressure used by `mhdFoam` to control div(B)"
        ),
    },
    parameters=[
        PDEParameter("Re", "Reynolds number mapped to nu = 1/Re"),
        PDEParameter("Rm", "Magnetic Reynolds number mapped to sigma = Rm"),
        PDEParameter("k", "Retained config degree parameter"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "magnetic_field": InputSpec(
            name="magnetic_field",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        ),
        "magnetic_field": BoundaryFieldSpec(
            name="magnetic_field",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the magnetic field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "magnetic_field": StateSpec(name="magnetic_field", shape="vector"),
        "magnetic_constraint": StateSpec(name="magnetic_constraint", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "magnetic_field": OutputSpec(
            name="magnetic_field",
            shape="vector",
            output_mode="components",
            source_name="magnetic_field",
        ),
        "magnetic_constraint": OutputSpec(
            name="magnetic_constraint",
            shape="scalar",
            output_mode="scalar",
            source_name="magnetic_constraint",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


@register_preset("mhd")
class MHDPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _MHD_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return OpenFOAMMHDProblem(self.spec, config)
