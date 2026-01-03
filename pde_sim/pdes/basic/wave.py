"""Wave equation."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("wave")
class WavePDE(MultiFieldPDEPreset):
    """Wave equation.

    The wave equation describes wave propagation:

        d²u/dt² = c² * laplace(u)

    We convert to first order system:
        du/dt = v
        dv/dt = c² * laplace(u)

    where u is displacement and v is velocity.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="wave",
            category="basic",
            description="Wave equation (second order)",
            equations={
                "u": "v",
                "v": "c**2 * laplace(u)",
            },
            parameters=[
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Wave speed",
                    min_value=0.5,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Hyperbolic PDE",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 1.0)
        c_sq = c * c

        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={
                "u": "v",
                "v": f"{c_sq} * laplace(u)",
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for wave equation."""
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("advection")
class AdvectionPDE(MultiFieldPDEPreset):
    """Advection-diffusion equation.

    Describes transport of a quantity by flow and diffusion:

        du/dt = D * laplace(u) - v · grad(u)

    where D is diffusion and v is velocity vector.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advection",
            category="basic",
            description="Advection-diffusion equation",
            equations={
                "u": "D * laplace(u) - vx * d_dx(u) - vy * d_dy(u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.05,
                    description="Diffusion coefficient",
                    min_value=0.0,
                    max_value=0.2,
                ),
                PDEParameter(
                    name="vx",
                    default=0.5,
                    description="Advection velocity in x",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="vy",
                    default=0.0,
                    description="Advection velocity in y",
                    min_value=-2.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Convection-diffusion equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.01)
        vx = parameters.get("vx", 1.0)
        vy = parameters.get("vy", 0.0)

        # Build equation string
        terms = []
        if D > 0:
            terms.append(f"{D} * laplace(u)")
        if vx != 0:
            terms.append(f"- {vx} * d_dx(u)")
        if vy != 0:
            terms.append(f"- {vy} * d_dy(u)")

        rhs = " ".join(terms) if terms else "0"

        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={"u": rhs},
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("inhomogeneous-wave")
class InhomogeneousWavePDE(MultiFieldPDEPreset):
    """Inhomogeneous wave equation with damping and source.

    Wave equation with additional terms:

        d²u/dt² = c² * laplace(u) - gamma * du/dt + source

    Converted to first-order system:
        du/dt = v
        dv/dt = c² * laplace(u) - gamma * v + source

    where:
        - u is displacement
        - v is velocity
        - c is wave speed
        - gamma is damping coefficient
        - source is a forcing term
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="inhomogeneous-wave",
            category="basic",
            description="Wave equation with damping and source",
            equations={
                "u": "v",
                "v": "c**2 * laplace(u) - gamma * v + source",
            },
            parameters=[
                PDEParameter(
                    name="c",
                    default=1.0,
                    description="Wave speed",
                    min_value=0.5,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Damping coefficient",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="source",
                    default=0.0,
                    description="Constant forcing term",
                    min_value=-10.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Damped wave equation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        c = parameters.get("c", 1.0)
        gamma = parameters.get("gamma", 0.1)
        source = parameters.get("source", 0.0)
        c_sq = c * c

        # Build v equation
        terms = [f"{c_sq} * laplace(u)"]
        if gamma > 0:
            terms.append(f"- {gamma} * v")
        if source != 0:
            if source > 0:
                terms.append(f"+ {source}")
            else:
                terms.append(f"- {abs(source)}")

        v_rhs = " ".join(terms)

        bc_spec = self._convert_bc(bc)
        return PDE(
            rhs={
                "u": "v",
                "v": v_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for inhomogeneous wave equation."""
        # u gets the specified initial condition
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        # v (velocity) starts at zero by default
        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])
