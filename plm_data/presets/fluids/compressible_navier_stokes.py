"""Smooth compressible Navier-Stokes preset using primitive variables."""

from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance
from plm_data.presets.fluids._openfoam import OpenFOAMCompressibleNavierStokesProblem
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

_DENSITY_BOUNDARY_OPERATORS = {
    "dirichlet": SCALAR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
}

_COMPRESSIBLE_NAVIER_STOKES_SPEC = PresetSpec(
    name="compressible_navier_stokes",
    category="fluids",
    description=(
        "Smooth ideal-gas compressible Navier-Stokes flow in primitive variables "
        "(density, velocity, temperature) with heat conduction."
    ),
    equations={
        "density": "d(rho)/dt + div(rho * u) = s_rho",
        "velocity": (
            "rho * (du/dt + (u.grad)u) + grad(p) = div(tau(u)) + f_u, "
            "p = gas_constant * rho * temperature"
        ),
        "temperature": (
            "rho*c_v*(dT/dt + u.grad(T)) + p*div(u) = "
            "tau(u):grad(u) + thermal_conductivity*laplacian(T) + s_T"
        ),
    },
    parameters=[
        PDEParameter("gas_constant", "Ideal-gas constant relating pressure to rho*T"),
        PDEParameter("c_v", "Specific heat at constant volume"),
        PDEParameter("mu", "Dynamic shear viscosity"),
        PDEParameter("bulk_viscosity", "Bulk viscosity coefficient"),
        PDEParameter("thermal_conductivity", "Thermal conductivity coefficient"),
        PDEParameter("k", "Retained config degree parameter"),
    ],
    inputs={
        "density": InputSpec(
            name="density",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
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
        "density": BoundaryFieldSpec(
            name="density",
            shape="scalar",
            operators=_DENSITY_BOUNDARY_OPERATORS,
            description="Boundary conditions for density.",
        ),
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for velocity.",
        ),
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for temperature.",
        ),
    },
    states={
        "density": StateSpec(name="density", shape="scalar"),
        "velocity": StateSpec(name="velocity", shape="vector"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
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
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2, 3],
)


@register_preset("compressible_navier_stokes")
class CompressibleNavierStokesPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _COMPRESSIBLE_NAVIER_STOKES_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return OpenFOAMCompressibleNavierStokesProblem(self.spec, config)
