"""Advanced Nonlinear Physics PDEs."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, ScalarPDEPreset, PDEMetadata, PDEParameter, SolverType
from .. import register_pde


@register_pde("lorenz")
class LorenzPDE(MultiFieldPDEPreset):
    """Diffusively coupled Lorenz system.

    The Lorenz equations with spatial diffusion coupling:

        dx/dt = Dx * laplace(x) + sigma * (y - x)
        dy/dt = Dy * laplace(y) + x * (rho - z) - y
        dz/dt = Dz * laplace(z) + x * y - beta * z

    where:
        - x, y, z are the Lorenz variables
        - sigma, rho, beta are the classic Lorenz parameters
        - Dx, Dy, Dz are diffusion coefficients

    Creates spatiotemporal chaos from coupled chaotic oscillators.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="lorenz",
            category="physics",
            description="Diffusively coupled Lorenz system",
            equations={
                "x": "Dx * laplace(x) + sigma * (y - x)",
                "y": "Dy * laplace(y) + x * (rho - z) - y",
                "z": "Dz * laplace(z) + x * y - beta * z",
            },
            parameters=[
                PDEParameter(
                    name="sigma",
                    default=10.0,
                    description="Lorenz sigma parameter",
                    min_value=1.0,
                    max_value=20.0,
                ),
                PDEParameter(
                    name="rho",
                    default=28.0,
                    description="Lorenz rho parameter",
                    min_value=1.0,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="beta",
                    default=8.0 / 3.0,
                    description="Lorenz beta parameter",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dx",
                    default=0.1,
                    description="Diffusion of x",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dy",
                    default=0.1,
                    description="Diffusion of y",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dz",
                    default=0.1,
                    description="Diffusion of z",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=3,
            field_names=["x", "y", "z"],
            reference="Lorenz (1963) attractor with diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        sigma = parameters.get("sigma", 10.0)
        rho = parameters.get("rho", 28.0)
        beta = parameters.get("beta", 8.0 / 3.0)
        Dx = parameters.get("Dx", 0.1)
        Dy = parameters.get("Dy", 0.1)
        Dz = parameters.get("Dz", 0.1)

        return PDE(
            rhs={
                "x": f"{Dx} * laplace(x) + {sigma} * (y - x)",
                "y": f"{Dy} * laplace(y) + x * ({rho} - z) - y",
                "z": f"{Dz} * laplace(z) + x * y - {beta} * z",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near Lorenz attractor."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.5)

        # Start near one fixed point with perturbations
        x_data = 1.0 + noise * np.random.randn(*grid.shape)
        y_data = 1.0 + noise * np.random.randn(*grid.shape)
        z_data = 1.0 + noise * np.random.randn(*grid.shape)

        x = ScalarField(grid, x_data)
        x.label = "x"
        y = ScalarField(grid, y_data)
        y.label = "y"
        z = ScalarField(grid, z_data)
        z.label = "z"

        return FieldCollection([x, y, z])


@register_pde("superlattice")
class SuperlatticePDE(ScalarPDEPreset):
    """Superlattice pattern formation.

    Swift-Hohenberg with hexagonal modulation for superlattice patterns:

        du/dt = epsilon * u - (1 + laplace)^2 * u + g2 * u^2 - u^3

    With additional modulation to promote superlattice structures.
    """

    @property
    def default_solver(self) -> SolverType:
        """Superlattice is stiff due to 4th order derivatives."""
        return "implicit"

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="superlattice",
            category="physics",
            description="Superlattice pattern formation",
            equations={"u": "epsilon * u - (1 + laplace)^2 * u + g2 * u^2 - u^3"},
            parameters=[
                PDEParameter(
                    name="epsilon",
                    default=0.1,
                    description="Control parameter",
                    min_value=-0.5,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="g2",
                    default=0.5,
                    description="Quadratic nonlinearity",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Superlattice patterns in nonlinear optics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        epsilon = parameters.get("epsilon", 0.1)
        g2 = parameters.get("g2", 0.5)

        # (1 + laplace)^2 = 1 + 2*laplace + laplace(laplace)
        return PDE(
            rhs={
                "u": f"{epsilon} * u - u - 2 * laplace(u) - laplace(laplace(u)) + {g2} * u**2 - u**3"
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial hexagonal seed pattern."""
        if ic_type in ("superlattice-default", "default"):
            np.random.seed(ic_params.get("seed"))
            amplitude = ic_params.get("amplitude", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 2 * np.pi, grid.shape[0]),
                np.linspace(0, 2 * np.pi, grid.shape[1]),
                indexing="ij",
            )

            # Hexagonal seed + noise
            k = 1.0
            hex_pattern = (
                np.cos(k * x)
                + np.cos(k * (x / 2 + y * np.sqrt(3) / 2))
                + np.cos(k * (x / 2 - y * np.sqrt(3) / 2))
            )
            data = amplitude * (hex_pattern / 3 + 0.1 * np.random.randn(*grid.shape))
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("oscillators")
class OscillatorsPDE(MultiFieldPDEPreset):
    """Coupled nonlinear oscillators (Van der Pol type).

    Spatially extended Van der Pol oscillator:

        dx/dt = Dx * laplace(x) + y
        dy/dt = Dy * laplace(y) + mu * (1 - x^2) * y - omega^2 * x

    where:
        - x is the position variable
        - y is the velocity variable
        - mu controls nonlinearity strength
        - omega is the natural frequency
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="oscillators",
            category="physics",
            description="Coupled Van der Pol oscillators",
            equations={
                "x": "Dx * laplace(x) + y",
                "y": "Dy * laplace(y) + mu * (1 - x^2) * y - omega^2 * x",
            },
            parameters=[
                PDEParameter(
                    name="mu",
                    default=1.0,
                    description="Nonlinearity strength",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="omega",
                    default=1.0,
                    description="Natural frequency",
                    min_value=0.1,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dx",
                    default=0.1,
                    description="Diffusion of x",
                    min_value=0.0,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="Dy",
                    default=0.1,
                    description="Diffusion of y",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ],
            num_fields=2,
            field_names=["x", "y"],
            reference="Van der Pol oscillator field",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        mu = parameters.get("mu", 1.0)
        omega = parameters.get("omega", 1.0)
        Dx = parameters.get("Dx", 0.1)
        Dy = parameters.get("Dy", 0.1)
        omega_sq = omega**2

        return PDE(
            rhs={
                "x": f"{Dx} * laplace(x) + y",
                "y": f"{Dy} * laplace(y) + {mu} * (1 - x**2) * y - {omega_sq} * x",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial oscillator states."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.5)

        x_data = noise * np.random.randn(*grid.shape)
        y_data = noise * np.random.randn(*grid.shape)

        x = ScalarField(grid, x_data)
        x.label = "x"
        y = ScalarField(grid, y_data)
        y.label = "y"

        return FieldCollection([x, y])


@register_pde("perona-malik")
class PeronaMalikPDE(ScalarPDEPreset):
    """Perona-Malik anisotropic diffusion.

    Image processing PDE for edge-preserving smoothing:

        du/dt = div(g(|grad(u)|) * grad(u))

    where g(s) = 1/(1 + (s/K)^2) is the diffusivity function.

    Simplified version:
        du/dt = D * laplace(u) - D * K * gradient_squared(u) / (K^2 + gradient_squared(u))
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="perona-malik",
            category="physics",
            description="Perona-Malik edge-preserving diffusion",
            equations={"u": "D * laplace(u) / (1 + gradient_squared(u) / K^2)"},
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Diffusion coefficient",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="K",
                    default=1.0,
                    description="Edge threshold",
                    min_value=0.1,
                    max_value=10.0,
                ),
            ],
            num_fields=1,
            field_names=["u"],
            reference="Perona & Malik (1990) anisotropic diffusion",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        K = parameters.get("K", 1.0)
        K_sq = K**2

        # Simplified: linear diffusion with edge-dependent reduction
        return PDE(
            rhs={"u": f"{D} * laplace(u) / (1 + gradient_squared(u) / {K_sq})"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial image-like pattern with edges."""
        if ic_type in ("perona-malik-default", "default"):
            np.random.seed(ic_params.get("seed"))

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            # Step pattern with noise (like a noisy image with edges)
            data = np.zeros(grid.shape)
            data[x > 0.3] = 0.5
            data[x > 0.7] = 1.0
            data[y > 0.5] += 0.3

            # Add noise
            data += 0.1 * np.random.randn(*grid.shape)
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("nonlinear-beams")
class NonlinearBeamsPDE(MultiFieldPDEPreset):
    """Nonlinear beam equation.

    Fourth-order wave equation with geometric nonlinearity:

        d²u/dt² = -D * laplace(laplace(u)) + alpha * laplace((du/dx)²)

    Converted to first-order system:
        du/dt = v
        dv/dt = -D * laplace(laplace(u)) + alpha * laplace(gradient_squared(u))
    """

    @property
    def default_solver(self) -> SolverType:
        """Nonlinear beams is stiff due to 4th order derivatives."""
        return "implicit"

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="nonlinear-beams",
            category="physics",
            description="Nonlinear beam/plate vibrations",
            equations={
                "u": "v",
                "v": "-D * laplace(laplace(u)) + alpha * laplace(gradient_squared(u))",
            },
            parameters=[
                PDEParameter(
                    name="D",
                    default=1.0,
                    description="Bending stiffness",
                    min_value=0.01,
                    max_value=10.0,
                ),
                PDEParameter(
                    name="alpha",
                    default=0.1,
                    description="Nonlinearity coefficient",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Nonlinear plate vibrations",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        D = parameters.get("D", 1.0)
        alpha = parameters.get("alpha", 0.1)

        return PDE(
            rhs={
                "u": "v",
                "v": f"-{D} * laplace(laplace(u)) + {alpha} * laplace(gradient_squared(u))",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial beam displacement."""
        u = create_initial_condition(grid, ic_type, ic_params)
        u.label = "u"

        v_data = np.zeros(grid.shape)
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("turing-wave")
class TuringWavePDE(MultiFieldPDEPreset):
    """Turing patterns with wave instability.

    Reaction-diffusion system exhibiting both Turing and wave instabilities:

        du/dt = Du * laplace(u) + f(u, v)
        dv/dt = Dv * laplace(v) + g(u, v)

    with kinetics chosen to produce oscillating patterns.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="turing-wave",
            category="physics",
            description="Turing-wave pattern interaction",
            equations={
                "u": "Du * laplace(u) + a - u + u^2 * v - gamma * u",
                "v": "Dv * laplace(v) + b - u^2 * v + gamma * u",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=5.0,
                    description="Inhibitor diffusion",
                    min_value=0.1,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Inhibitor production",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="gamma",
                    default=0.1,
                    description="Cross-coupling",
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Turing-Hopf interaction patterns",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 5.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)
        gamma = parameters.get("gamma", 0.1)

        return PDE(
            rhs={
                "u": f"{Du} * laplace(u) + {a} - u + u**2 * v - {gamma} * u",
                "v": f"{Dv} * laplace(v) + {b} - u**2 * v + {gamma} * u",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial state near steady state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("advecting-patterns")
class AdvectingPatternsPDE(MultiFieldPDEPreset):
    """Turing patterns with advection.

    Reaction-diffusion with flow:

        du/dt = Du * laplace(u) - v_flow * d_dx(u) + f(u,v)
        dv/dt = Dv * laplace(v) - v_flow * d_dx(v) + g(u,v)

    Shows how flow affects pattern formation.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="advecting-patterns",
            category="physics",
            description="Advected Turing patterns",
            equations={
                "u": "Du * laplace(u) - v_flow * d_dx(u) + a - u + u^2 * v",
                "v": "Dv * laplace(v) - v_flow * d_dx(v) + b - u^2 * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=5.0,
                    description="Inhibitor diffusion",
                    min_value=0.1,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="v_flow",
                    default=0.5,
                    description="Advection velocity",
                    min_value=0.0,
                    max_value=5.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Inhibitor production",
                    min_value=0.0,
                    max_value=2.0,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pattern formation in flows",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 5.0)
        v_flow = parameters.get("v_flow", 0.5)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)

        u_rhs = f"{Du} * laplace(u) + {a} - u + u**2 * v"
        v_rhs = f"{Dv} * laplace(v) + {b} - u**2 * v"

        if v_flow != 0:
            u_rhs = f"{Du} * laplace(u) - {v_flow} * d_dx(u) + {a} - u + u**2 * v"
            v_rhs = f"{Dv} * laplace(v) - {v_flow} * d_dx(v) + {b} - u**2 * v"

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
        """Create initial state."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])


@register_pde("growing-domains")
class GrowingDomainsPDE(MultiFieldPDEPreset):
    """Reaction-diffusion on effectively growing domain.

    Simulated domain growth via dilution terms:

        du/dt = Du * laplace(u) + f(u,v) - rho * u
        dv/dt = Dv * laplace(v) + g(u,v) - rho * v

    where rho represents the dilution due to domain growth.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="growing-domains",
            category="physics",
            description="Patterns on growing domains",
            equations={
                "u": "Du * laplace(u) + a - u + u^2 * v - rho * u",
                "v": "Dv * laplace(v) + b - u^2 * v - rho * v",
            },
            parameters=[
                PDEParameter(
                    name="Du",
                    default=0.1,
                    description="Activator diffusion",
                    min_value=0.01,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Dv",
                    default=5.0,
                    description="Inhibitor diffusion",
                    min_value=0.1,
                    max_value=50.0,
                ),
                PDEParameter(
                    name="a",
                    default=0.1,
                    description="Activator production",
                    min_value=0.0,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="b",
                    default=0.9,
                    description="Inhibitor production",
                    min_value=0.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="rho",
                    default=0.01,
                    description="Growth/dilution rate",
                    min_value=0.0,
                    max_value=0.5,
                ),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Pattern formation on growing domains",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        Du = parameters.get("Du", 0.1)
        Dv = parameters.get("Dv", 5.0)
        a = parameters.get("a", 0.1)
        b = parameters.get("b", 0.9)
        rho = parameters.get("rho", 0.01)

        u_rhs = f"{Du} * laplace(u) + {a} - u + u**2 * v - {rho} * u"
        v_rhs = f"{Dv} * laplace(v) + {b} - u**2 * v - {rho} * v"

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
        """Create initial state with seed pattern."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.05)

        u_data = 0.5 + noise * np.random.randn(*grid.shape)
        v_data = 0.5 + noise * np.random.randn(*grid.shape)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])
