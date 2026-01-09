"""Main simulation orchestrator."""

import re
import time
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


def _get_next_folder_number(base_path: Path, preset: str, overwrite: bool = False) -> str:
    """Find the next available folder number for a preset.

    Args:
        base_path: Base output directory.
        preset: Name of the PDE preset.
        overwrite: If True, return the last used number instead of incrementing.

    Returns:
        The next folder number as a zero-padded string (e.g., "001").
    """
    preset_dir = base_path / preset
    if not preset_dir.exists():
        return "001"

    # Find all existing numbered folders
    existing_numbers = []
    for item in preset_dir.iterdir():
        if item.is_dir():
            match = re.match(r"^(\d+)$", item.name)
            if match:
                existing_numbers.append(int(match.group(1)))

    if not existing_numbers:
        return "001"

    max_number = max(existing_numbers)
    if overwrite:
        return f"{max_number:03d}"
    return f"{max_number + 1:03d}"


class SimulationRunner:
    """Main simulation orchestrator.

    Coordinates PDE setup, time stepping, and output generation.
    """

    def __init__(
        self,
        config: SimulationConfig,
        output_dir: Path | str | None = None,
        sim_id: str | None = None,
        overwrite: bool = False,
    ):
        """Initialize the simulation runner.

        Args:
            config: Simulation configuration.
            output_dir: Override for output directory.
            sim_id: Override for simulation ID (auto-generated if None).
            overwrite: If True, overwrite the last numbered folder instead of creating a new one.
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else config.output.path
        self.sim_id = sim_id or str(uuid.uuid4())

        # Generate run name with incremental number (zero-padded, e.g., "001")
        folder_number = _get_next_folder_number(self.output_dir, config.preset, overwrite)
        self.run_name = folder_number
        # Full folder path: {preset}/{number}
        self.folder_name = f"{config.preset}/{folder_number}"

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
            bc_config=config.bc,
        )

        # Create PDE
        params = self.preset.get_default_parameters()
        params.update(config.parameters)
        self.preset.validate_parameters(params)

        # Warn about unused parameters in config
        known_param_names = {p.name for p in self.preset.metadata.parameters}
        provided_param_names = set(config.parameters.keys())
        unused_params = provided_param_names - known_param_names
        if unused_params:
            warnings.warn(
                f"Unused parameters in config for '{config.preset}': {sorted(unused_params)}. "
                f"Known parameters are: {sorted(known_param_names)}",
                UserWarning,
                stacklevel=2,
            )

        self.parameters = params

        self.pde = self.preset.create_pde(
            parameters=params,
            bc=config.bc,  # Pass full BoundaryConfig to support per-field BCs
            grid=self.grid,
        )

        # Create initial state
        # Pass parameters and bc for PDEs that need them (e.g., plate equation)
        self.state = self.preset.create_initial_state(
            grid=self.grid,
            ic_type=config.init.type,
            ic_params=config.init.params,
            parameters=params,
            bc=config.bc,
        )

        # Determine field configurations for output
        field_configs = config.output.get_field_configs()

        # If no fields specified, default to all fields from PDE metadata
        if not field_configs:
            field_configs = [
                (name, config.output.colormap)
                for name in self.preset.metadata.field_names
            ]

        # Output management
        self.output_manager = OutputManager(
            base_path=self.output_dir,
            folder_name=self.folder_name,
            colormap=config.output.colormap,
            field_configs=field_configs,
            save_array=config.output.save_array,
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
            "scipy": "scipy",
        }
        return solver_map.get(self.config.solver.lower(), "euler")

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Execute the simulation and return metadata.

        Args:
            verbose: Whether to print progress information.

        Returns:
            Complete simulation metadata dictionary.
        """
        # Calculate save interval from num_frames
        # We want exactly num_frames output frames (including t=0 and t=t_end)
        num_frames = self.config.output.num_frames
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        save_interval = self.config.t_end / (num_frames - 1)

        if verbose:
            print(f"Starting simulation: {self.config.preset}")
            print(f"  Resolution: {self.config.resolution}x{self.config.resolution}")
            print(f"  Final time: {self.config.t_end}, dt: {self.config.dt}")
            print(f"  Frames: {num_frames} (interval: {save_interval:.4f})")
            print(f"  Solver: {self.config.solver}")
            print(f"  Backend: {self.config.backend}")
            # Only show adaptive info for solvers that support it
            if self._get_solver_name() != "implicit" and self.config.adaptive:
                print(f"  Adaptive: True (tolerance: {self.config.tolerance})")
            print(f"  Output: {self.output_manager.output_dir}")

        # Create storage for capturing frames
        storage = MemoryStorage()

        # Run simulation using py-pde's solve method
        # interrupts parameter determines the interval for saving
        tracker = storage.tracker(interrupts=save_interval)

        # Build solver kwargs - only explicit solvers support adaptive stepping
        solver_name = self._get_solver_name()
        solve_kwargs = {
            "t_range": self.config.t_end,
            "dt": self.config.dt,
            "solver": solver_name,
            "tracker": tracker if not verbose else ["progress", tracker],
            "backend": self.config.backend,
        }

        # Only pass adaptive/tolerance for solvers that support it
        # (implicit solver does not support adaptive stepping)
        if solver_name != "implicit":
            solve_kwargs["adaptive"] = self.config.adaptive
            solve_kwargs["tolerance"] = self.config.tolerance

        # Time the simulation
        wall_clock_start = time.perf_counter()
        result = self.pde.solve(self.state, **solve_kwargs)
        wall_clock_duration = time.perf_counter() - wall_clock_start

        # Capture solver diagnostics (including adaptive dt statistics)
        solver_diagnostics = getattr(self.pde, "diagnostics", {}).get("solver", {})

        # Save frames from storage
        if verbose:
            print(f"  Saving {len(storage)} frames...")

        # Two-pass approach for consistent colorscale:
        # 1. Compute global min/max across all frames per field
        all_fields = [field for _, field in storage.items()]
        all_times = list(storage.times)

        for field_name, _ in self.output_manager.field_configs:
            self.output_manager.compute_range_for_field(all_fields, field_name)

        # 2. Save frames with the pre-computed range
        for frame_index, (t, field) in enumerate(storage.items()):
            self.output_manager.save_all_fields(field, frame_index, t)

        # 3. Save trajectory array if requested (per field)
        if self.output_manager.save_array:
            for field_name, _ in self.output_manager.field_configs:
                self.output_manager.save_trajectory_array(all_fields, all_times, field_name)
            if verbose:
                print(f"  Saved trajectory arrays")

        # Generate and save metadata
        total_time = storage.times[-1] if storage.times else self.config.t_end

        metadata = create_metadata(
            sim_id=self.sim_id,
            preset_name=self.config.preset,
            preset_metadata=self.preset.metadata,
            config=self.config,
            total_time=total_time,
            frame_annotations=self.output_manager.get_frame_annotations(),
            solver_diagnostics=solver_diagnostics,
            wall_clock_duration=wall_clock_duration,
        )

        self.output_manager.save_metadata(metadata)

        if verbose:
            print(f"Simulation complete!")
            print(f"  Generated {len(storage)} frames")
            print(f"  Total simulation time: {total_time:.4f}")
            # Report adaptive dt statistics if available
            dt_stats = solver_diagnostics.get("dt_statistics")
            if dt_stats and solver_diagnostics.get("dt_adaptive"):
                print(f"  Adaptive dt: {dt_stats['min']:.3g} .. {dt_stats['max']:.3g} "
                      f"(mean: {dt_stats['mean']:.3g}, steps: {dt_stats['count']})")

        # Add folder_name to returned metadata (for CLI use, not saved to file)
        metadata["folder_name"] = self.folder_name
        return metadata


def run_from_config(
    config_path: Path | str,
    output_dir: Path | str | None = None,
    seed: int | None = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a simulation from a config file.

    Args:
        config_path: Path to YAML configuration file.
        output_dir: Override for output directory.
        seed: Override for random seed.
        verbose: Whether to print progress.
        overwrite: If True, overwrite the last numbered folder instead of creating a new one.

    Returns:
        Simulation metadata dictionary.
    """
    config = load_config(config_path)

    # Override seed if provided
    if seed is not None:
        config.seed = seed

    runner = SimulationRunner(config, output_dir=output_dir, overwrite=overwrite)
    return runner.run(verbose=verbose)
