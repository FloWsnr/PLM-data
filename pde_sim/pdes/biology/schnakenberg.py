"""Schnakenberg reaction-diffusion system for Turing patterns."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("schnakenberg")
class SchnakenbergPDE(MultiFieldPDEPreset):
    """Schnakenberg reaction-diffusion system.

    The Schnakenberg model produces Turing patterns:

        du/dt = Du * laplace(u) + a - u + u²v
        dv/dt = Dv * laplace(v) + b - u²v

    where:
        - u, v are chemical concentrations
        - Du, Dv are diffusion coefficients (Dv >> Du for patterns)
        - a, b are feed rates

    For Turing instability: Dv/Du > 1 (typically ~10-100)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="schnakenberg",
            category="biology",
            description="Schnakenberg Turing pattern system",
            equations={
                "u": "Du * laplace(u) + a - u + u**2 * v",
                "v": "Dv * laplace(v) + b - u**2 * v",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Feed rate for u",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Feed rate for v",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Diffusion of u (activator)",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=100.0,
                    description="Diffusion of v (inhibitor)",
                    min_value=1.0,
                    max_value=1000.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Turing pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 100.0)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {a} - u + u**2 * v",
                "v": f"{Dv} * laplace(v) + {b} - u**2 * v",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near the homogeneous steady state with perturbation."""
        # Steady state: u* = a + b, v* = b / (a + b)²
        a = ic_params.get("a", 0.1)
        b = ic_params.get("b", 0.9)
        noise = ic_params.get("noise", 0.01)

        u_ss = a + b
        v_ss = b / (u_ss ** 2)

        # Add small random perturbation
        np.random.seed(ic_params.get("seed"))
        u_data = u_ss + noise * np.random.randn(*grid.shape)
        v_data = v_ss + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("brusselator")
class BrusselatorPDE(MultiFieldPDEPreset):
    """Brusselator reaction-diffusion system.

    The Brusselator model for chemical oscillations and patterns:

        du/dt = Du * laplace(u) + a - (b+1)*u + u²v
        dv/dt = Dv * laplace(v) + b*u - u²v

    Exhibits both oscillations and Turing patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="brusselator",
            category="biology",
            description="Brusselator oscillating reaction-diffusion",
            equations={
                "u": "Du * laplace(u) + a - (b + 1) * u + u**2 * v",
                "v": "Dv * laplace(v) + b * u - u**2 * v",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Parameter a",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="b",
                    default=3.0,
                    description="Parameter b",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Diffusion of u",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=8.0,
                    description="Diffusion of v",
                    min_value=0.1,
                    max_value=100.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Brusselator chemical oscillator",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 1.0)
        b = parameters.get("b", 3.0)
        Du = parameters.get("Du", 1.0)
        Dv = parameters.get("Dv", 8.0)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{Dv} * laplace(v) + {b} * u - u**2 * v",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near homogeneous steady state."""
        a = ic_params.get("a", 1.0)
        b = ic_params.get("b", 3.0)
        noise = ic_params.get("noise", 0.1)

        # Steady state: u* = a, v* = b/a
        u_ss = a
        v_ss = b / a

        np.random.seed(ic_params.get("seed"))
        u_data = u_ss + noise * np.random.randn(*grid.shape)
        v_data = v_ss + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("fisher-kpp")
class FisherKPPPDE(ScalarPDEPreset):
    """Fisher-KPP equation for population dynamics.

    The Fisher-KPP equation describes population growth with diffusion:

        du/dt = D * laplace(u) + r * u * (1 - u/K)

    where:
        - u is population density
        - D is diffusion coefficient
        - r is growth rate
        - K is carrying capacity

    Exhibits traveling wave solutions.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="fisher-kpp",
            category="biology",
            description="Fisher-KPP population dynamics",
            equations={
                "u": "D * laplace(u) + r * u * (1 - u / K)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Carrying capacity",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Population invasion fronts",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        r = parameters.get("r", 1.0)
        K = parameters.get("K", 1.0)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r} * u * (1 - u / {K})"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )
