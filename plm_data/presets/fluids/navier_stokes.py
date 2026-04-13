"""Incompressible Navier-Stokes preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.fluids._openfoam import OpenFOAMNavierStokesProblem
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)


_NAVIER_STOKES_SPEC = PresetSpec(
    name="navier_stokes",
    category="fluids",
    description="Incompressible Navier-Stokes equations.",
    equations={
        "velocity": "du/dt + (u.grad)u = -grad(p) + (1/Re)*laplacian(u)",
        "pressure": "div(u) = 0",
    },
    parameters=[
        PDEParameter("Re", "Reynolds number"),
        PDEParameter("k", "Polynomial degree parameter retained in the config"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
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
        )
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
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
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


@register_preset("navier_stokes")
class NavierStokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _NAVIER_STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return OpenFOAMNavierStokesProblem(self.spec, config)
