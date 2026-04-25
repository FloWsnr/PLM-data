"""Elasticity PDE spec."""

from plm_data.pdes.metadata import VECTOR_STANDARD_BOUNDARY_OPERATORS

from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDESpec,
    StateSpec,
)

_ELASTICITY_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "neumann": VECTOR_STANDARD_BOUNDARY_OPERATORS["neumann"],
}

PDE_SPEC = PDESpec(
    name="elasticity",
    category="basic",
    description=(
        "Transient small-strain isotropic linear elasticity with Rayleigh "
        "damping, vector displacement dynamics, and a derived von Mises stress."
    ),
    equations={
        "displacement": "density * d²u/dt² + C(du/dt) - div(σ(u)) = f",
        "velocity": "v = du/dt",
    },
    parameters=[
        PDEParameter(
            "young_modulus",
            "Young's modulus",
            hard_min=0.0,
            sampling_min=4.5,
            sampling_max=6.5,
        ),
        PDEParameter(
            "poisson_ratio",
            "Poisson ratio",
            hard_min=-1.0,
            hard_max=0.5,
            sampling_min=0.22,
            sampling_max=0.34,
        ),
        PDEParameter(
            "density",
            "Mass density",
            hard_min=0.0,
            sampling_min=1.2,
            sampling_max=1.8,
        ),
        PDEParameter(
            "eta_mass",
            "Rayleigh mass-damping coefficient",
            hard_min=0.0,
            sampling_min=0.006,
            sampling_max=0.014,
        ),
        PDEParameter(
            "eta_stiffness",
            "Rayleigh stiffness-damping coefficient",
            hard_min=0.0,
            sampling_min=0.0003,
            sampling_max=0.0008,
        ),
    ],
    inputs={
        "displacement": InputSpec(
            name="displacement",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "forcing": InputSpec(
            name="forcing",
            shape="vector",
            allow_source=True,
            allow_initial_condition=False,
        ),
    },
    boundary_fields={
        "displacement": BoundaryFieldSpec(
            name="displacement",
            shape="vector",
            operators=_ELASTICITY_BOUNDARY_OPERATORS,
            description=(
                "Displacement boundary conditions. Use Dirichlet for clamped/"
                "prescribed motion and Neumann for traction-free or imposed traction "
                "boundaries."
            ),
        )
    },
    states={
        "displacement": StateSpec(name="displacement", shape="vector"),
        "velocity": StateSpec(name="velocity", shape="vector"),
    },
    outputs={
        "displacement": OutputSpec(
            name="displacement",
            shape="vector",
            output_mode="components",
            source_name="displacement",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "von_mises": OutputSpec(
            name="von_mises",
            shape="scalar",
            output_mode="scalar",
            source_name="von_mises",
            source_kind="derived",
        ),
    },
    static_fields=[],
    supported_dimensions=[2],
)

__all__ = ["PDE_SPEC"]
