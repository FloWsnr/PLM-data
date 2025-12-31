"""Main simulation orchestrator."""

import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pde import MemoryStorage

from .config import SimulationConfig, load_config
from .output import OutputManager, create_metadata
from ..boundaries import create_grid_with_bc
from ..pdes import get_pde_preset

# Valid backend options
VALID_BACKENDS = ("auto", "numpy", "numba")


class SimulationRunner:
    """Main simulation orchestrator.

    Coordinates PDE setup, time stepping, and output generation.
    """

    def __init__(
        self,
        config: SimulationConfig,
        output_dir: Path | str | None = None,
        sim_id: str | None = None,
    ):
        """Initialize the simulation runner.

        Args:
            config: Simulation configuration.
            output_dir: Override for output directory.
            sim_id: Override for simulation ID (auto-generated if None).
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else config.output.path
        self.sim_id = sim_id or str(uuid.uuid4())

        # Validate backend
        if config.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{config.backend}'. "
                f"Must be one of: {', '.join(VALID_BACKENDS)}"
            )

        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)

        # Initialize components
        self.preset = get_pde_preset(config.preset)

        # Create grid with appropriate boundary conditions
        self.grid = create_grid_with_bc(
            resolution=config.resolution,
            domain_size=config.domain_size,
            bc_config={"x": config.bc.x, "y": config.bc.y},
        )

        # Create PDE
        params = self.preset.get_default_parameters()
        params.update(config.parameters)
        self.preset.validate_parameters(params)
        self.parameters = params

        self.pde = self.preset.create_pde(
            parameters=params,
            bc={"x": config.bc.x, "y": config.bc.y},
            grid=self.grid,
        )

        # Create initial state
        self.state = self.preset.create_initial_state(
            grid=self.grid,
            ic_type=config.init.type,
            ic_params=config.init.params,
        )

        # Output management
        self.output_manager = OutputManager(
            base_path=self.output_dir,
            sim_id=self.sim_id,
            colormap=config.output.colormap,
            field_to_plot=config.output.field_to_plot,
        )

    def _get_solver_name(self) -> str:
        """Get the solver name for py-pde."""
        solver_map = {
            "euler": "euler",
            "rk4": "runge-kutta",
            "rk": "runge-kutta",
            "runge-kutta": "runge-kutta",
            "implicit": "implicit",
            "implicit-euler": "implicit",
        }
        return solver_map.get(self.config.solver.lower(), "euler")

    def _check_stability(self) -> None:
        """Check numerical stability conditions and warn if violated.

        Issues a RuntimeWarning if the time step exceeds the CFL stability limit
        for diffusion-type equations.
        """
        dx = self.config.domain_size / self.config.resolution
        dt = self.config.dt

        # Collect all diffusion-like coefficients from parameters
        diffusion_coeffs = []
        for key in ("D", "Du", "Dv", "Dw", "M", "diffusivity", "nu"):
            if key in self.parameters:
                val = self.parameters[key]
                if isinstance(val, (int, float)) and val > 0:
                    diffusion_coeffs.append(val)

        if not diffusion_coeffs:
            return

        D_max = max(diffusion_coeffs)

        # CFL condition for 2D diffusion: dt < dx^2 / (4 * D)
        dt_cfl = dx**2 / (4 * D_max)

        if dt > dt_cfl:
            warnings.warn(
                f"Time step dt={dt} exceeds CFL stability limit "
                f"dt_max={dt_cfl:.6f} for diffusion coefficient D={D_max}. "
                f"Consider reducing dt or using adaptive=True.",
                RuntimeWarning,
                stacklevel=3,
            )

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Execute the simulation and return metadata.

        Args:
            verbose: Whether to print progress information.

        Returns:
            Complete simulation metadata dictionary.
        """
        # Check numerical stability
        self._check_stability()

        if verbose:
            print(f"Starting simulation: {self.config.preset}")
            print(f"  Resolution: {self.config.resolution}x{self.config.resolution}")
            print(f"  Timesteps: {self.config.timesteps}, dt: {self.config.dt}")
            print(f"  Solver: {self.config.solver}")
            print(f"  Backend: {self.config.backend}")
            if self.config.adaptive:
                print(f"  Adaptive: True (tolerance: {self.config.tolerance})")
            print(f"  Output: {self.output_manager.output_dir}")

        # Calculate time parameters
        t_end = self.config.timesteps * self.config.dt
        frames_per_save = self.config.output.frames_per_save
        save_interval = frames_per_save * self.config.dt

        # Create storage for capturing frames
        storage = MemoryStorage()

        # Run simulation using py-pde's solve method
        # interrupts parameter determines the interval for saving
        tracker = storage.tracker(interrupts=save_interval)

        result = self.pde.solve(
            self.state,
            t_range=t_end,
            dt=self.config.dt,
            solver=self._get_solver_name(),
            tracker=tracker if not verbose else ["progress", tracker],
            backend=self.config.backend,
            adaptive=self.config.adaptive,
            tolerance=self.config.tolerance,
        )

        # Save frames from storage
        if verbose:
            print(f"  Saving {len(storage)} frames...")

        for frame_index, (time, field) in enumerate(storage.items()):
            self.output_manager.save_frame(field, frame_index, time)

        # Generate and save metadata
        total_time = storage.times[-1] if storage.times else t_end

        metadata = create_metadata(
            sim_id=self.sim_id,
            preset_name=self.config.preset,
            preset_metadata=self.preset.metadata,
            config=self.config,
            total_time=total_time,
            frame_annotations=self.output_manager.get_frame_annotations(),
        )

        self.output_manager.save_metadata(metadata)

        if verbose:
            print(f"Simulation complete!")
            print(f"  Generated {len(storage)} frames")
            print(f"  Total simulation time: {total_time:.4f}")

        return metadata


def run_from_config(
    config_path: Path | str,
    output_dir: Path | str | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run a simulation from a config file.

    Args:
        config_path: Path to YAML configuration file.
        output_dir: Override for output directory.
        seed: Override for random seed.
        verbose: Whether to print progress.

    Returns:
        Simulation metadata dictionary.
    """
    config = load_config(config_path)

    # Override seed if provided
    if seed is not None:
        config.seed = seed

    runner = SimulationRunner(config, output_dir=output_dir)
    return runner.run(verbose=verbose)
