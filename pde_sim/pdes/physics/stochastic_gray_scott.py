"""Stochastic Gray-Scott reaction-diffusion system with multiplicative noise."""

from typing import Any

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField
from pde.pdes.base import SDEBase

from ..base import MultiFieldPDEPreset, PDEMetadata, PDEParameter
from .. import register_pde


class MultiplicativeNoiseGrayScottPDE(SDEBase):
    """Gray-Scott PDE with true multiplicative noise on the u field.

    Implements:
        du/dt = laplace(u) + u^2*v - (a+b)*u + sigma*u*dW
        dv/dt = D*laplace(v) - u^2*v + a*(1-v)

    where dW is spatiotemporal white noise and the noise term sigma*u*dW
    is multiplicative (Itô interpretation).

    This class extends py-pde's SDEBase to properly implement multiplicative
    noise by overriding the noise_realization method.
    """

    def __init__(
        self,
        a: float = 0.037,
        b: float = 0.04,
        D: float = 4.0,
        sigma: float = 0.0,
        bc="auto_periodic_neumann",
        rng: np.random.Generator | None = None,
    ):
        """Initialize the stochastic Gray-Scott PDE.

        Args:
            a: Feed rate
            b: Kill/removal rate
            D: Diffusion ratio (v diffuses D times faster than u)
            sigma: Noise intensity for multiplicative noise on u
            bc: Boundary conditions
            rng: Random number generator
        """
        # Initialize SDEBase with noise=1 as a marker that we have noise
        # The actual noise scaling is handled in noise_realization
        super().__init__(noise=sigma if sigma > 0 else 0, rng=rng)
        self.a = a
        self.b = b
        self.D = D
        self.sigma = sigma
        self.bc = bc

    @property
    def is_sde(self) -> bool:
        """Check if this is a stochastic PDE."""
        return self.sigma > 0

    def evolution_rate(self, state: FieldCollection, t: float = 0) -> FieldCollection:
        """Evaluate the deterministic part of the RHS.

        Args:
            state: FieldCollection with [u, v] fields
            t: Current time

        Returns:
            FieldCollection with evolution rates for [u, v]
        """
        u = state[0]
        v = state[1]

        # Deterministic Gray-Scott equations
        # du/dt = laplace(u) + u^2*v - (a+b)*u
        # dv/dt = D*laplace(v) - u^2*v + a*(1-v)
        u_rate = u.laplace(bc=self.bc) + u.data**2 * v.data - (self.a + self.b) * u.data
        v_rate = (
            self.D * v.laplace(bc=self.bc) - u.data**2 * v.data + self.a * (1 - v.data)
        )

        # Create result fields
        result_u = ScalarField(state.grid, u_rate)
        result_u.label = "u"
        result_v = ScalarField(state.grid, v_rate)
        result_v.label = "v"

        return FieldCollection([result_u, result_v])

    def noise_realization(
        self, state: FieldCollection, t: float = 0, *, label: str = "Noise realization"
    ) -> FieldCollection:
        """Return a realization of the multiplicative noise.

        For multiplicative noise sigma*u*dW, we return sigma*u*xi where xi
        is standard Gaussian white noise. The sqrt(dt) scaling is handled
        by the solver.

        Args:
            state: Current state FieldCollection with [u, v]
            t: Current time
            label: Label for result

        Returns:
            FieldCollection with noise realizations for [u, v]
        """
        if not self.is_sde:
            # No noise - return zeros
            result = state.copy()
            for field in result:
                field.data.fill(0)
            if label:
                result.label = label
            return result

        u = state[0]
        grid = state.grid

        # Generate spatiotemporal white noise with proper physical scaling
        # For proper scaling: noise ~ 1/sqrt(cell_volume) at each grid point
        cell_volume = np.prod(grid.discretization)
        noise_scale = 1.0 / np.sqrt(cell_volume)

        # Generate standard normal noise
        white_noise = self.rng.standard_normal(grid.shape) * noise_scale

        # Multiplicative noise on u: sigma * u * white_noise
        u_noise_data = self.sigma * u.data * white_noise

        # No noise on v
        v_noise_data = np.zeros(grid.shape)

        # Create result fields
        result_u = ScalarField(grid, u_noise_data)
        result_u.label = "u"
        result_v = ScalarField(grid, v_noise_data)
        result_v.label = "v"

        result = FieldCollection([result_u, result_v])
        if label:
            result.label = label
        return result

    def make_noise_realization_numba(self, state: FieldCollection):
        """Create a numba-compiled function for noise realization.

        Args:
            state: Example state for extracting grid info

        Returns:
            Callable for generating noise realizations
        """
        if not self.is_sde:

            def noise_realization(state_data, t):
                return None

            return noise_realization

        sigma = self.sigma
        grid = state.grid
        data_shape = state.data.shape
        cell_volume = float(np.prod(grid.discretization))
        noise_scale = 1.0 / np.sqrt(cell_volume)

        # Get slice info for u field (first field in collection)
        u_slice_start = state._slices[0].start
        u_slice_stop = state._slices[0].stop
        u_shape = state[0].data.shape

        def noise_realization(state_data, t):
            """Generate multiplicative noise realization."""
            out = np.zeros(data_shape)

            # Get current u field data
            u_data = state_data[u_slice_start:u_slice_stop].reshape(u_shape)

            # Generate white noise and apply multiplicative scaling
            white_noise = np.random.randn(*u_shape) * noise_scale
            u_noise = sigma * u_data * white_noise

            # Place noise in output array (only u field)
            out[u_slice_start:u_slice_stop] = u_noise.ravel()

            return out

        return noise_realization


@register_pde("stochastic-gray-scott")
class StochasticGrayScottPDE(MultiFieldPDEPreset):
    """Stochastic Gray-Scott reaction-diffusion system.

    Based on the visualpde.com formulation with multiplicative noise:

        du/dt = laplace(u) + u^2*v - (a+b)*u + sigma * u * dW
        dv/dt = D * laplace(v) - u^2*v + a*(1-v)

    where:
        - u is the activator concentration
        - v is the substrate concentration
        - a is the feed rate
        - b is the kill/removal rate
        - D is the diffusion ratio
        - sigma is the noise intensity
        - dW is spatiotemporal white noise (Wiener process increment)

    Key phenomena in stochastic reaction-diffusion systems:
        - Noise-induced patterns: Patterns emerge that would not exist deterministically
        - Stochastic resonance: Intermediate noise levels optimize pattern formation
        - Stochastic extinction: Very strong noise destroys all patterns

    IMPORTANT: The stochastic terms only work correctly with Euler timestepping!

    Reference: Biancalani et al. (2010), Woolley et al. (2011)
    """

    @property
    def metadata(self) -> PDEMetadata:
        return PDEMetadata(
            name="stochastic-gray-scott",
            category="physics",
            description="Stochastic Gray-Scott with noise-induced patterns",
            equations={
                "u": "laplace(u) + u**2 * v - (a + b) * u + sigma * u * dW",
                "v": "D * laplace(v) - u**2 * v + a * (1 - v)",
            },
            parameters=[
                PDEParameter("a", "Feed rate"),
                PDEParameter("b", "Kill/removal rate"),
                PDEParameter("D", "Diffusion ratio (v diffuses D times faster than u)"),
                PDEParameter("sigma", "Noise intensity (0 = deterministic)"),
            ],
            num_fields=2,
            field_names=["u", "v"],
            reference="Biancalani et al. (2010) Stochastic Turing patterns",
            supported_dimensions=[1, 2, 3],
        )

    def create_pde(
        self,
        parameters: dict[str, float],
        bc: dict[str, Any],
        grid: CartesianGrid,
    ) -> MultiplicativeNoiseGrayScottPDE:
        """Create the Stochastic Gray-Scott PDE system.

        Args:
            parameters: Dictionary with a, b, D, sigma.
            bc: Boundary condition specification.
            grid: The computational grid.

        Returns:
            Configured PDE instance with true multiplicative noise.
        """
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.04)
        D = parameters.get("D", 4.0)
        sigma = parameters.get("sigma", 0.0)

        return MultiplicativeNoiseGrayScottPDE(
            a=a,
            b=b,
            D=D,
            sigma=sigma,
            bc=self._convert_bc(bc),
        )

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create initial state for Stochastic Gray-Scott.

        Default initialization: small circular patch of u on uniform v=1 background.
        """
        # Check for per-field specifications
        if "u" in ic_params and isinstance(ic_params["u"], dict):
            return super().create_initial_state(grid, ic_type, ic_params)

        if ic_type in ("stochastic-gray-scott-default", "default"):
            return self._default_stochastic_init(grid, ic_params)

        # Fallback to parent implementation
        return super().create_initial_state(grid, ic_type, ic_params)

    def _default_stochastic_init(
        self,
        grid: CartesianGrid,
        params: dict[str, Any],
    ) -> FieldCollection:
        """Create default initialization with small circular patch of u."""
        # Get domain info
        x_bounds = grid.axes_bounds[0]
        y_bounds = grid.axes_bounds[1]
        Lx = x_bounds[1] - x_bounds[0]
        Ly = y_bounds[1] - y_bounds[0]

        # Create coordinates
        x = np.linspace(x_bounds[0], x_bounds[1], grid.shape[0])
        y = np.linspace(y_bounds[0], y_bounds[1], grid.shape[1])
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Center of domain
        cx = x_bounds[0] + Lx / 2
        cy = y_bounds[0] + Ly / 2

        # Small circular patch
        r = params.get("perturbation_radius", 0.05) * min(Lx, Ly)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r

        # Initialize u = 0, v = 1 everywhere
        u_data = np.zeros(grid.shape)
        v_data = np.ones(grid.shape)

        # Add small patch of u in center
        u_data[mask] = 0.5

        # Add very small noise
        noise = params.get("noise", 0.001)
        seed = params.get("seed")
        rng = np.random.default_rng(seed)
        u_data += noise * rng.standard_normal(grid.shape)
        v_data += noise * rng.standard_normal(grid.shape)

        # Clip to valid range
        u_data = np.clip(u_data, 0, 1)
        v_data = np.clip(v_data, 0, 1)

        u = ScalarField(grid, u_data)
        u.label = "u"
        v = ScalarField(grid, v_data)
        v.label = "v"

        return FieldCollection([u, v])

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted."""
        a = parameters.get("a", 0.037)
        b = parameters.get("b", 0.04)
        D = parameters.get("D", 4.0)
        sigma = parameters.get("sigma", 0.0)

        return {
            "u": f"laplace(u) + u**2 * v - ({a} + {b}) * u + {sigma} * u * dW",
            "v": f"{D} * laplace(v) - u**2 * v + {a} * (1 - v)",
        }
