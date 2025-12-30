"""Gierer-Meinhardt activator-inhibitor system."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("gierer-meinhardt")
class GiererMeinhardtPDE(MultiFieldPDEPreset):
    """Gierer-Meinhardt activator-inhibitor system.

    A classic model for biological pattern formation:

        du/dt = Du * laplace(u) + rho * (u^2 / v) - mu_u * u + rho_u
        dv/dt = Dv * laplace(v) + rho * u^2 - mu_v * v

    where:
        - u is the activator concentration
        - v is the inhibitor concentration
        - Du, Dv are diffusion coefficients (Dv >> Du for patterns)
        - rho is the production rate
        - mu_u, mu_v are decay rates
        - rho_u is the basal activator production

    For Turing patterns: Dv/Du >> 1 (typically ~10-100)

    Reference: Gierer & Meinhardt (1972)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="gierer-meinhardt",
            category="biology",
            description="Gierer-Meinhardt activator-inhibitor pattern formation",
            equations={
                "u": "Du * laplace(u) + rho * u**2 / v - mu_u * u + rho_u",
                "v": "Dv * laplace(v) + rho * u**2 - mu_v * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.01,
                    description="Activator diffusion coefficient",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=1.0,
                    description="Inhibitor diffusion coefficient",
                    min_value=0.1,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="rho",
                    default=1.0,
                    description="Production rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="mu_u",
                    default=0.1,
                    description="Activator decay rate",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="mu_v",
                    default=0.1,
                    description="Inhibitor decay rate",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="rho_u",
                    default=0.01,
                    description="Basal activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Gierer & Meinhardt (1972) Turing patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.01)
        Dv = parameters.get("Dv", 1.0)
        rho = parameters.get("rho", 1.0)
        mu_u = parameters.get("mu_u", 0.1)
        mu_v = parameters.get("mu_v", 0.1)
        rho_u = parameters.get("rho_u", 0.01)

        # Build equation strings
        u_rhs = f"{Du} * laplace(u) + {rho} * u**2 / (v + 1e-10) - {mu_u} * u + {rho_u}"
        v_rhs = f"{Dv} * laplace(v) + {rho} * u**2 - {mu_v} * v"

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        return PDE(
            rhs={
                "u": u_rhs,
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
        """Create initial state near steady state with perturbation."""
        noise = ic_params.get("noise", 0.01)

        # Initial values near steady state (approximate)
        u0 = ic_params.get("u0", 1.0)
        v0 = ic_params.get("v0", 1.0)

        np.random.seed(ic_params.get("seed"))
        u_data = u0 * (1 + noise * np.random.randn(*grid.shape))
        v_data = v0 * (1 + noise * np.random.randn(*grid.shape))

        # Ensure positive values
        u_data = np.maximum(u_data, 0.01)
        v_data = np.maximum(v_data, 0.01)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("keller-segel")
class KellerSegelPDE(MultiFieldPDEPreset):
    """Keller-Segel chemotaxis model.

    Describes cell movement in response to chemical signals:

        du/dt = Du * laplace(u) - chi * div(u * grad(c))
        dc/dt = Dc * laplace(c) + alpha * u - beta * c

    where:
        - u is the cell density
        - c is the chemoattractant concentration
        - Du, Dc are diffusion coefficients
        - chi is the chemotactic sensitivity
        - alpha is the chemoattractant production rate
        - beta is the chemoattractant decay rate

    This model can exhibit blow-up (aggregation) in finite time.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="keller-segel",
            category="biology",
            description="Keller-Segel chemotaxis model",
            equations={
                "u": "Du * laplace(u) - chi * div(u * gradient(c))",
                "c": "Dc * laplace(c) + alpha * u - beta * c",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=1.0,
                    description="Cell diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dc",
                    default=1.0,
                    description="Chemoattractant diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="chi",
                    default=1.0,
                    description="Chemotactic sensitivity",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Chemoattractant production rate",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="beta",
                    default=1.0,
                    description="Chemoattractant decay rate",
                    min_value=0.01,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "c"],
            reference="Keller & Segel (1970) chemotaxis",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 1.0)
        Dc = parameters.get("Dc", 1.0)
        chi = parameters.get("chi", 1.0)
        alpha = parameters.get("alpha", 1.0)
        beta = parameters.get("beta", 1.0)

        # Note: py-pde uses "gradient" not "grad"
        # The divergence of u*grad(c) requires careful handling
        # Using the approximation: div(u*grad(c)) ~ u*laplace(c) + grad(u).grad(c)
        u_rhs = f"{Du} * laplace(u) - {chi} * (u * laplace(c) + inner(gradient(u), gradient(c)))"
        c_rhs = f"{Dc} * laplace(c) + {alpha} * u - {beta} * c"

        bc_spec = "periodic" if bc.get("x") == "periodic" else "no-flux"

        return PDE(
            rhs={
                "u": u_rhs,
                "c": c_rhs,
            },
            bc=bc_spec,
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with localized cell density."""
        from pde_sim.initial_conditions import create_initial_condition

        # Default: use gaussian blobs for cells, uniform for chemoattractant
        if ic_type in ("keller-segel-default", "default"):
            np.random.seed(ic_params.get("seed"))

            # Cells - gaussian blobs
            u = create_initial_condition(
                grid, "gaussian-blobs", {"num_blobs": 3, "amplitude": 1.0, "width": 0.1}
            )
            u.label = "u"

            # Chemoattractant - start uniform
            c = ScalarField(grid, np.ones(grid.shape) * 0.1)
            c.label = "c"

            return FieldCollection([u, c])

        # Use parent implementation for other IC types
        return super().create_initial_state(grid, ic_type, ic_params)
