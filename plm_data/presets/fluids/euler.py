"""OpenFOAM-backed compressible Euler preset."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.fluids._openfoam import OpenFOAMEulerProblem
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    BoundaryOperatorSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_EULER_SCALAR_BOUNDARY_OPERATORS = {
    "dirichlet": SCALAR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": SCALAR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

_EULER_VECTOR_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": VECTOR_STANDARD_BOUNDARY_OPERATORS["neumann"],
    "periodic": VECTOR_STANDARD_BOUNDARY_OPERATORS["periodic"],
    "slip": BoundaryOperatorSpec(
        name="slip",
        value_shape=None,
        description="Inviscid slip wall boundary condition.",
    ),
}

_EULER_SPEC = PresetSpec(
    name="euler",
    category="fluids",
    description=(
        "Compressible ideal-gas Euler flow in primitive variables "
        "(density, velocity, pressure), solved with the OpenFOAM shockFluid "
        "backend."
    ),
    equations={
        "density": "d(rho)/dt + div(rho * u) = 0",
        "velocity": (
            "du/dt + (u.grad)u + grad(p) / rho = 0, "
            "with p = gas_constant * rho * temperature"
        ),
        "pressure": (
            "dp/dt + u.grad(p) + gamma * p * div(u) = 0, "
            "where gamma = 1 + gas_constant / c_v"
        ),
    },
    parameters=[
        PDEParameter("gas_constant", "Ideal-gas constant relating p, rho, and T"),
        PDEParameter("c_v", "Specific heat at constant volume"),
        PDEParameter("k", "Retained config degree parameter"),
    ],
    inputs={
        "density": InputSpec(
            name="density",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "pressure": InputSpec(
            name="pressure",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "density": BoundaryFieldSpec(
            name="density",
            shape="scalar",
            operators=_EULER_SCALAR_BOUNDARY_OPERATORS,
            description="Density boundary conditions.",
        ),
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=_EULER_VECTOR_BOUNDARY_OPERATORS,
            description="Velocity boundary conditions.",
        ),
        "pressure": BoundaryFieldSpec(
            name="pressure",
            shape="scalar",
            operators=_EULER_SCALAR_BOUNDARY_OPERATORS,
            description="Pressure boundary conditions.",
        ),
    },
    states={
        "density": StateSpec(name="density", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
    },
    outputs={
        "density": OutputSpec(
            name="density",
            shape="scalar",
            output_mode="scalar",
            source_name="density",
        ),
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
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


@register_preset("euler")
class EulerPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _EULER_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return OpenFOAMEulerProblem(self.spec, config)
