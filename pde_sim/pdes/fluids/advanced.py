"""Advanced Fluid Dynamics PDEs."""

from typing import Any

import numpy as np
from pde import PDE, CartesianGrid, FieldCollection, ScalarField

from pde_sim.initial_conditions import create_initial_condition

from ..base import MultiFieldPDEPreset, ScalarPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


@register_pde("navier-stokes")
class NavierStokesPDE(MultiFieldPDEPreset):
    """2D Navier-Stokes in vorticity-stream function formulation.

    Full vorticity equation with advection:

        dw/dt = nu * laplace(w) - u * d_dx(w) - v * d_dy(w)

    where velocity (u, v) is derived from stream function psi.

    Simplified version with explicit velocity fields:
        dw/dt = nu * laplace(w) - (d_dy(psi) * d_dx(w) - d_dx(psi) * d_dy(w))
        laplace(psi) = -w

    For simulation, we use a simplified model where we solve the
    vorticity transport with diffusion and a simple shear flow.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="navier-stokes",
            category="fluids",
            description="2D Navier-Stokes vorticity equation",
            equations={
                "w": "nu * laplace(w) - ux * d_dx(w) - uy * d_dy(w)",
            },
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="ux",
                    default=0.1,
                    description="Background velocity x",
                    min_value=-2.0,
                    max_value=2.0,
                ),
                PDEParameter(
                    name="uy",
                    default=0.0,
                    description="Background velocity y",
                    min_value=-2.0,
                    max_value=2.0,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="2D incompressible Navier-Stokes",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)
        ux = parameters.get("ux", 0.1)
        uy = parameters.get("uy", 0.0)

        # Vorticity transport with background flow
        rhs = f"{nu} * laplace(w)"
        if ux != 0:
            rhs += f" - {ux} * d_dx(w)"
        if uy != 0:
            rhs += f" - {uy} * d_dy(w)"

        return PDE(
            rhs={"w": rhs},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create initial vorticity distribution."""
        if ic_type in ("navier-stokes-default", "default", "vortex-pair"):
            # Counter-rotating vortex pair
            x0_1, y0_1 = ic_params.get("x1", 0.35), ic_params.get("y1", 0.5)
            x0_2, y0_2 = ic_params.get("x2", 0.65), ic_params.get("y2", 0.5)
            strength = ic_params.get("strength", 10.0)
            radius = ic_params.get("radius", 0.1)

            x, y = np.meshgrid(
                np.linspace(0, 1, grid.shape[0]),
                np.linspace(0, 1, grid.shape[1]),
                indexing="ij",
            )

            r1_sq = (x - x0_1) ** 2 + (y - y0_1) ** 2
            r2_sq = (x - x0_2) ** 2 + (y - y0_2) ** 2

            w1 = strength * np.exp(-r1_sq / (2 * radius**2))
            w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

            data = w1 + w2
            return ScalarField(grid, data)

        return create_initial_condition(grid, ic_type, ic_params)


@register_pde("thermal-convection")
class ThermalConvectionPDE(MultiFieldPDEPreset):
    """Rayleigh-Bénard thermal convection.

    Coupled temperature and stream function equations:

        dT/dt = kappa * laplace(T) - u * d_dx(T) - v * d_dy(T)
        dw/dt = nu * laplace(w) + Ra * d_dx(T)

    where:
        - T is temperature
        - w is vorticity
        - Ra is Rayleigh number (buoyancy driving)
        - kappa is thermal diffusivity
        - nu is kinematic viscosity

    Simplified version with explicit coupling.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="thermal-convection",
            category="fluids",
            description="Rayleigh-Bénard convection",
            equations={
                "T": "kappa * laplace(T)",
                "w": "nu * laplace(w) + Ra * d_dx(T)",
            },
            parameters=[
                PDEParameter(
                    name="kappa",
                    default=0.01,
                    description="Thermal diffusivity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="Ra",
                    default=100.0,
                    description="Rayleigh number",
                    min_value=1.0,
                    max_value=10000.0,
                ),
            ],
            num_fields=2,
            field_names=["T", "w"],
            reference="Rayleigh-Bénard convection rolls",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        kappa = parameters.get("kappa", 0.01)
        nu = parameters.get("nu", 0.01)
        Ra = parameters.get("Ra", 100.0)

        return PDE(
            rhs={
                "T": f"{kappa} * laplace(T)",
                "w": f"{nu} * laplace(w) + {Ra} * d_dx(T)",
            },
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> FieldCollection:
        """Create initial temperature gradient with perturbation."""
        np.random.seed(ic_params.get("seed"))
        noise = ic_params.get("noise", 0.1)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Linear temperature profile (hot bottom, cold top) with perturbation
        T_data = 1.0 - y + noise * np.random.randn(*grid.shape)

        # Small initial vorticity
        w_data = noise * np.random.randn(*grid.shape)

        T = ScalarField(grid, T_data)
        T.label = "T"
        w = ScalarField(grid, w_data)
        w.label = "w"

        return FieldCollection([T, w])


@register_pde("method-of-images")
class MethodOfImagesPDE(ScalarPDEPreset):
    """Potential flow near a wall using method of images concept.

    Vortex dynamics with wall reflection symmetry:

        dw/dt = nu * laplace(w)

    Initial condition uses mirror vortices to simulate wall effect.
    The dynamics evolve the vorticity with viscous diffusion.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="method-of-images",
            category="fluids",
            description="Vortex near wall (method of images)",
            equations={"w": "nu * laplace(w)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.01,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=1.0,
                ),
                PDEParameter(
                    name="wall_distance",
                    default=0.2,
                    description="Distance of vortex from wall",
                    min_value=0.05,
                    max_value=0.4,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Potential flow method of images",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.01)

        return PDE(
            rhs={"w": f"{nu} * laplace(w)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create vortex pair near wall (real + image)."""
        wall_dist = ic_params.get("wall_distance", 0.2)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.08)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Wall at y = 0, vortex at y = wall_dist
        # Image vortex at y = -wall_dist (reflected, opposite sign)
        y_vortex = wall_dist
        y_image = -wall_dist  # Outside domain, but affects field

        # Real vortex
        r1_sq = (x - 0.5) ** 2 + (y - y_vortex) ** 2
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))

        # Image vortex (opposite circulation)
        r2_sq = (x - 0.5) ** 2 + (y - y_image) ** 2
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        # Add mirror on the other side for periodic BC
        r3_sq = (x - 0.5) ** 2 + (y - (1.0 - wall_dist)) ** 2
        w3 = strength * np.exp(-r3_sq / (2 * radius**2))
        r4_sq = (x - 0.5) ** 2 + (y - (1.0 + wall_dist)) ** 2
        w4 = -strength * np.exp(-r4_sq / (2 * radius**2))

        data = w1 + w2 + w3 + w4
        return ScalarField(grid, data)


@register_pde("dipoles")
class DipolesPDE(ScalarPDEPreset):
    """Vortex dipole dynamics.

    Two co-rotating vortices that move together as a dipole:

        dw/dt = nu * laplace(w)

    The dipole structure creates self-induced motion.
    With diffusion, the dipoles gradually decay.
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="dipoles",
            category="fluids",
            description="Vortex dipole motion and decay",
            equations={"w": "nu * laplace(w)"},
            parameters=[
                PDEParameter(
                    name="nu",
                    default=0.005,
                    description="Kinematic viscosity",
                    min_value=0.001,
                    max_value=0.5,
                ),
                PDEParameter(
                    name="separation",
                    default=0.15,
                    description="Vortex separation distance",
                    min_value=0.05,
                    max_value=0.3,
                ),
            ],
            num_fields=1,
            field_names=["w"],
            reference="Vortex dipole dynamics",
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> PDE:
        nu = parameters.get("nu", 0.005)

        return PDE(
            rhs={"w": f"{nu} * laplace(w)"},
            bc="periodic" if bc.get("x") == "periodic" else "no-flux",
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField:
        """Create vortex dipole (two opposite-sign vortices)."""
        separation = ic_params.get("separation", 0.15)
        strength = ic_params.get("strength", 10.0)
        radius = ic_params.get("radius", 0.05)

        x, y = np.meshgrid(
            np.linspace(0, 1, grid.shape[0]),
            np.linspace(0, 1, grid.shape[1]),
            indexing="ij",
        )

        # Dipole centered at (0.3, 0.5), moving to the right
        cx, cy = 0.3, 0.5

        # Two vortices separated vertically
        y1 = cy + separation / 2
        y2 = cy - separation / 2

        r1_sq = (x - cx) ** 2 + (y - y1) ** 2
        r2_sq = (x - cx) ** 2 + (y - y2) ** 2

        # Opposite circulations create rightward motion
        w1 = strength * np.exp(-r1_sq / (2 * radius**2))
        w2 = -strength * np.exp(-r2_sq / (2 * radius**2))

        data = w1 + w2
        return ScalarField(grid, data)
