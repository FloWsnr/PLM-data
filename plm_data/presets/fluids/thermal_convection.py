"""Thermal convection (Rayleigh-Benard) preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.fluids._openfoam import OpenFOAMThermalConvectionProblem
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)


_THERMAL_CONVECTION_SPEC = PresetSpec(
    name="thermal_convection",
    category="fluids",
    description="Thermal convection using the Boussinesq Rayleigh-Benard equations.",
    equations={
        "velocity": (
            "du/dt + (u_prev.grad)u = -grad(p) + sqrt(Pr/Ra)*laplacian(u) + T*e_v"
        ),
        "pressure": "div(u) = 0",
        "temperature": ("dT/dt + u_prev.grad(T) = (1/sqrt(Ra*Pr))*laplacian(T) + f_T"),
    },
    parameters=[
        PDEParameter("Ra", "Rayleigh number"),
        PDEParameter("Pr", "Prandtl number"),
        PDEParameter("k", "Retained config degree parameter"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "temperature": InputSpec(
            name="temperature",
            shape="scalar",
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
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the temperature field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
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
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


@register_preset("thermal_convection")
class ThermalConvectionPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _THERMAL_CONVECTION_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return OpenFOAMThermalConvectionProblem(self.spec, config)
