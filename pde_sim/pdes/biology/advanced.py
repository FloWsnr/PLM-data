"""Advanced Mathematical Biology PDEs."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("cyclic-competition")
class CyclicCompetitionPDE(MultiFieldPDEPreset):
    """Cyclic competition (rock-paper-scissors) model.

    Three-species competition where each species dominates one other:
    A beats B, B beats C, C beats A (like rock-paper-scissors).

        du/dt = Du * laplace(u) + u * (1 - u - v - w) - alpha * u * v
        dv/dt = Dv * laplace(v) + v * (1 - u - v - w) - alpha * v * w
        dw/dt = Dw * laplace(w) + w * (1 - u - v - w) - alpha * w * u

    where:
        - u, v, w are species densities
        - D is diffusion coefficient
        - alpha is competition strength

    Exhibits spiral wave patterns and biodiversity maintenance.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cyclic-competition",
            category="biology",
            description="Rock-paper-scissors cyclic competition",
            equations={
                "u": "D * laplace(u) + u * (1 - u - v - w) - alpha * u * v",
                "v": "D * laplace(v) + v * (1 - u - v - w) - alpha * v * w",
                "w": "D * laplace(w) + w * (1 - u - v - w) - alpha * w * u",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Competition strength",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["u", "v", "w"],
            reference="May-Leonard cyclic competition model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        alpha = parameters.get("alpha", 1.0)

        return PDE(
            rhs={
                "u": f"{D} * laplace(u) + u * (1 - u - v - w) - {alpha} * u * v",
                "v": f"{D} * laplace(v) + v * (1 - u - v - w) - {alpha} * v * w",
                "w": f"{D} * laplace(w) + w * (1 - u - v - w) - {alpha} * w * u",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with three species in different regions."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Initialize with random small values
        u_data = 0.33 + noise * np.random.randn(*grid.shape)
        v_data = 0.33 + noise * np.random.randn(*grid.shape)
        w_data = 0.33 + noise * np.random.randn(*grid.shape)

        # Ensure non-negative
        u_data = np.clip(u_data, 0.01, 1.0)
        v_data = np.clip(v_data, 0.01, 1.0)
        w_data = np.clip(w_data, 0.01, 1.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([u, v, w])


@register_pde("vegetation")
class VegetationPDE(MultiFieldPDEPreset):
    """Klausmeier vegetation model for semi-arid ecosystems.

    Models water-vegetation dynamics on sloped terrain:

        dw/dt = a - w - w * n^2 + v * d_dx(w)
        dn/dt = Dn * laplace(n) + w * n^2 - m * n

    where:
        - w is water density
        - n is vegetation (plant) density
        - a is rainfall rate
        - v is water flow velocity (downhill)
        - m is plant mortality rate
        - Dn is plant dispersal

    Produces banded vegetation patterns on hillslopes.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="vegetation",
            category="biology",
            description="Klausmeier vegetation-water dynamics",
            equations={
                "w": "a - w - w * n**2 + v * d_dx(w)",
                "n": "Dn * laplace(n) + w * n**2 - m * n",
            },
            parameters=[
                PDEParameter(
                    name="a",
                    default=2.0,
                    description="Rainfall rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="v",
                    default=1.0,
                    description="Water flow velocity",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="m",
                    default=0.45,
                    description="Plant mortality rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="Dn",
                    default=0.01,
                    description="Plant dispersal coefficient",
                    min_value=0.001,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["w", "n"],
            reference="Klausmeier (1999) vegetation patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        a = parameters.get("a", 2.0)
        v = parameters.get("v", 1.0)
        m = parameters.get("m", 0.45)
        Dn = parameters.get("Dn", 0.01)

        # Water equation with advection
        w_rhs = f"{a} - w - w * n**2"
        if v != 0:
            w_rhs += f" + {v} * d_dx(w)"

        return PDE(
            rhs={
                "w": w_rhs,
                "n": f"{Dn} * laplace(n) + w * n**2 - {m} * n",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state with water and vegetation."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        # Uniform water, patchy vegetation
        w_data = 2.0 * np.ones(grid.shape)
        n_data = 0.5 + noise * np.random.randn(*grid.shape)
        n_data = np.clip(n_data, 0.01, 2.0)

        w = ScalarField(grid, w_data)
        w.label = "w"
        n = ScalarField(grid, n_data)
        n.label = "n"

        return FieldCollection([w, n])


@register_pde("cross-diffusion")
class CrossDiffusionPDE(MultiFieldPDEPreset):
    """Cross-diffusion system (Shigesada-Kawasaki-Teramoto model).

    Two species with density-dependent diffusion:

        du/dt = laplace((d1 + a11*u + a12*v) * u) + u * (r1 - b1*u - c1*v)
        dv/dt = laplace((d2 + a21*u + a22*v) * v) + v * (r2 - b2*v - c2*u)

    Simplified version:
        du/dt = D1 * laplace(u) + alpha * laplace(u*v) + u * (1 - u - v)
        dv/dt = D2 * laplace(v) + beta * laplace(u*v) + v * (1 - u - v)

    Cross-diffusion can produce patterns even without Turing instability.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="cross-diffusion",
            category="biology",
            description="Cross-diffusion pattern formation",
            equations={
                "u": "D1 * laplace(u) + alpha * laplace(u*v) + u * (1 - u - v)",
                "v": "D2 * laplace(v) + beta * laplace(u*v) + v * (1 - u - v)",
            },
            parameters=[
                PDEParameter(
                    name="D1",
                    default=0.1,
                    description="Diffusion of species u",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="D2",
                    default=0.1,
                    description="Diffusion of species v",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=1.0,
                    description="Cross-diffusion coefficient for u",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="beta",
                    default=0.0,
                    description="Cross-diffusion coefficient for v",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Shigesada-Kawasaki-Teramoto cross-diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D1 = parameters.get("D1", 0.1)
        D2 = parameters.get("D2", 0.1)
        alpha = parameters.get("alpha", 1.0)
        beta = parameters.get("beta", 0.0)

        u_rhs = f"{D1} * laplace(u) + {alpha} * laplace(u*v) + u * (1 - u - v)"
        v_rhs = f"{D2} * laplace(v) + {beta} * laplace(u*v) + v * (1 - u - v)"

        return PDE(
            rhs={"u": u_rhs, "v": v_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state for cross-diffusion system."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)
        u_data = np.clip(u_data, 0.01, 1.0)
        v_data = np.clip(v_data, 0.01, 1.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("immunotherapy")
class ImmunotherapyPDE(MultiFieldPDEPreset):
    """Tumor-immune interaction model.

    Simplified tumor-immune dynamics with therapy:

        dT/dt = Dt * laplace(T) + rT * T * (1 - T/K) - k * T * I
        dI/dt = Di * laplace(I) + s + rI * T * I / (a + T) - dI * I

    where:
        - T is tumor cell density
        - I is immune cell density
        - rT is tumor growth rate
        - k is kill rate by immune cells
        - s is immune cell source (therapy)
        - rI is immune recruitment rate
        - dI is immune cell death rate
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="immunotherapy",
            category="biology",
            description="Tumor-immune interaction dynamics",
            equations={
                "T": "Dt * laplace(T) + rT * T * (1 - T/K) - k * T * I",
                "I": "Di * laplace(I) + s + rI * T * I / (a + T) - dI * I",
            },
            parameters=[
                PDEParameter(
                    name="Dt",
                    default=0.01,
                    description="Tumor diffusion",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Di",
                    default=0.1,
                    description="Immune cell diffusion",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="rT",
                    default=1.0,
                    description="Tumor growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Tumor carrying capacity",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="k",
                    default=1.0,
                    description="Immune kill rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="s",
                    default=0.1,
                    description="Immune source (therapy)",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="rI",
                    default=0.5,
                    description="Immune recruitment rate",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Half-saturation constant",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="dI",
                    default=0.3,
                    description="Immune death rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["T", "I"],
            reference="Tumor-immune system dynamics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Dt = parameters.get("Dt", 0.01)
        Di = parameters.get("Di", 0.1)
        rT = parameters.get("rT", 1.0)
        K = parameters.get("K", 1.0)
        k = parameters.get("k", 1.0)
        s = parameters.get("s", 0.1)
        rI = parameters.get("rI", 0.5)
        a = parameters.get("a", 0.1)
        dI = parameters.get("dI", 0.3)

        T_rhs = f"{Dt} * laplace(T) + {rT} * T * (1 - T / {K}) - {k} * T * I"
        I_rhs = f"{Di} * laplace(I) + {s} + {rI} * T * I / ({a} + T) - {dI} * I"

        return PDE(
            rhs={"T": T_rhs, "I": I_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial tumor with some immune cells."""
        np.random.seed(ic_params.get("seed"))

        # Small tumor in center
        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )
        r_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
        T_data = 0.5 * np.exp(-r_sq / 0.02)

        # Uniform low immune presence
        I_data = 0.1 * np.ones(grid.shape)

        T = ScalarField(grid, T_data)
        T.label = "T"
        I = ScalarField(grid, I_data)
        I.label = "I"

        return FieldCollection([T, I])


@register_pde("harsh-environment")
class HarshEnvironmentPDE(ScalarPDEPreset):
    """Population in harsh environment with Allee effect.

    Population dynamics with strong Allee effect:

        du/dt = D * laplace(u) + r * u * (u - theta) * (1 - u)

    where:
        - u is population density (normalized)
        - D is diffusion coefficient
        - r is growth rate
        - theta is Allee threshold (below which population declines)

    Exhibits bistability: extinction or survival depending on initial density.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="harsh-environment",
            category="biology",
            description="Allee effect in harsh environment",
            equations={"u": "D * laplace(u) + r * u * (u - theta) * (1 - u)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.001,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="theta",
                    default=0.2,
                    description="Allee threshold",
                    min_value=0.0,
                    max_value=0.5,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Strong Allee effect model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        r = parameters.get("r", 1.0)
        theta = parameters.get("theta", 0.2)

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r} * u * (u - {theta}) * (1 - u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial population patch."""
        if ic_type in ("harsh-environment-default", "default"):
            # Population patch above Allee threshold
            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )
            r_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
            data = 0.8 * np.exp(-r_sq / 0.05)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("bacteria-flow")
class BacteriaFlowPDE(MultiFieldPDEPreset):
    """Bacteria with chemotaxis in flowing medium.

    Bacteria moving up chemical gradients with advection:

        db/dt = Db * laplace(b) - chi * div(b * grad(c)) - v * d_dx(b) + r * b
        dc/dt = Dc * laplace(c) - k * c + s * b

    where:
        - b is bacteria density
        - c is chemoattractant concentration
        - chi is chemotaxis strength
        - v is flow velocity
        - r is bacteria growth rate
        - k is chemical decay rate
        - s is chemical production by bacteria
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="bacteria-flow",
            category="biology",
            description="Bacteria chemotaxis with flow",
            equations={
                "b": "Db * laplace(b) - chi * div(b * grad(c)) - v * d_dx(b) + r * b",
                "c": "Dc * laplace(c) - k * c + s * b",
            },
            parameters=[
                PDEParameter(
                    name="Db",
                    default=0.1,
                    description="Bacteria diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dc",
                    default=1.0,
                    description="Chemical diffusion",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="chi",
                    default=1.0,
                    description="Chemotaxis strength",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="v",
                    default=0.5,
                    description="Flow velocity",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="r",
                    default=0.1,
                    description="Bacteria growth rate",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="k",
                    default=0.1,
                    description="Chemical decay rate",
                    min_value=0.01,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="s",
                    default=0.1,
                    description="Chemical production rate",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["b", "c"],
            reference="Chemotaxis with advection",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Db = parameters.get("Db", 0.1)
        Dc = parameters.get("Dc", 1.0)
        chi = parameters.get("chi", 1.0)
        v = parameters.get("v", 0.5)
        r = parameters.get("r", 0.1)
        k = parameters.get("k", 0.1)
        s = parameters.get("s", 0.1)

        # Simplified chemotaxis (linearized around uniform c)
        b_rhs = f"{Db} * laplace(b) - {chi} * (d_dx(b) * d_dx(c) + d_dy(b) * d_dy(c))"
        if v != 0:
            b_rhs += f" - {v} * d_dx(b)"
        if r != 0:
            b_rhs += f" + {r} * b"

        c_rhs = f"{Dc} * laplace(c) - {k} * c + {s} * b"

        return PDE(
            rhs={"b": b_rhs, "c": c_rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial bacteria colony with chemical."""
        np.random.seed(ic_params.get("seed"))

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Bacteria colony
        r_sq = (x - 0.3) ** 2 + (y - 0.5) ** 2
        b_data = np.exp(-r_sq / 0.02)

        # Low uniform chemical
        c_data = 0.1 * np.ones(grid.shape)

        b = ScalarField(grid, b_data)
        b.label = "b"
        c = ScalarField(grid, c_data)
        c.label = "c"

        return FieldCollection([b, c])


@register_pde("heterogeneous")
class HeterogeneousPDE(ScalarPDEPreset):
    """Reaction-diffusion with spatially varying parameters.

    Fisher-KPP with space-dependent growth rate:

        du/dt = D * laplace(u) + r(x,y) * u * (1 - u)

    where r(x,y) varies sinusoidally across the domain,
    creating regions of favorable and unfavorable habitat.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="heterogeneous",
            category="biology",
            description="Reaction-diffusion with spatial heterogeneity",
            equations={"u": "D * laplace(u) + r(x,y) * u * (1 - u)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="r_mean",
                    default=1.0,
                    description="Mean growth rate",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="r_amp",
                    default=0.5,
                    description="Growth rate amplitude",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="freq",
                    default=2.0,
                    description="Spatial frequency of variation",
                    min_value=1.0,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Heterogeneous environment model",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        r_mean = parameters.get("r_mean", 1.0)
        r_amp = parameters.get("r_amp", 0.5)
        freq = parameters.get("freq", 2.0)

        # r(x,y) = r_mean + r_amp * sin(freq * 2 * pi * x)
        # Use x coordinate directly in py-pde
        r_expr = f"({r_mean} + {r_amp} * sin({freq} * 2 * 3.14159 * x))"

        return PDE(
            rhs={"u": f"{D} * laplace(u) + {r_expr} * u * (1 - u)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )


@register_pde("topography")
class TopographyPDE(MultiFieldPDEPreset):
    """Population dynamics on varying terrain.

    Movement biased by terrain slope:

        du/dt = D * laplace(u) - alpha * div(u * grad(h)) + r * u * (1 - u)

    where h is a fixed elevation field affecting movement.
    Simplified: use advection toward low points.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="topography",
            category="biology",
            description="Population dynamics on terrain",
            equations={
                "u": "D * laplace(u) - alpha * (d_dx(u) * hx + d_dy(u) * hy) + r * u * (1 - u)",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=0.1,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.5,
                    description="Terrain influence strength",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="r",
                    default=1.0,
                    description="Growth rate",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="hx",
                    default=0.1,
                    description="Terrain slope in x",
                    min_value=-1.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="hy",
                    default=0.0,
                    description="Terrain slope in y",
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Population on heterogeneous terrain",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 0.1)
        alpha = parameters.get("alpha", 0.5)
        r = parameters.get("r", 1.0)
        hx = parameters.get("hx", 0.1)
        hy = parameters.get("hy", 0.0)

        # Advection-diffusion with terrain
        rhs = f"{D} * laplace(u) - {alpha} * ({hx} * d_dx(u) + {hy} * d_dy(u)) + {r} * u * (1 - u)"

        return PDE(
            rhs={"u": rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("turing-conditions")
class TuringConditionsPDE(MultiFieldPDEPreset):
    """Demonstration of Turing pattern conditions.

    Two-species activator-inhibitor system with parameters
    chosen to satisfy Turing instability conditions:

        du/dt = Du * laplace(u) + f(u,v)
        dv/dt = Dv * laplace(v) + g(u,v)

    where f = a * u - b * v and g = c * u - d * v (linearized)
    Full: f = u * (a - u) - u * v,  g = u * v - d * v

    Turing conditions require Dv >> Du and specific kinetics.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-conditions",
            category="biology",
            description="Turing instability demonstration",
            equations={
                "u": "Du * laplace(u) + u * (a - u) - u * v",
                "v": "Dv * laplace(v) + u * v - d * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion (small)",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=10.0,
                    description="Inhibitor diffusion (large)",
                    min_value=1.0,
                    max_value=100.0,
                ),
                PDEParameter(
                    name="a",
                    default=1.0,
                    description="Activator growth rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="d",
                    default=0.5,
                    description="Inhibitor decay rate",
                    min_value=0.1,
                    max_value=5.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Turing (1952) pattern formation",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 10.0)
        a = parameters.get("a", 1.0)
        d = parameters.get("d", 0.5)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + u * ({a} - u) - u * v",
                "v": f"{Dv} * laplace(v) + u * v - {d} * v",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create small perturbation around steady state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        # Near uniform with small perturbations
        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)
        u_data = np.clip(u_data, 0.01, 2.0)
        v_data = np.clip(v_data, 0.01, 2.0)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])
