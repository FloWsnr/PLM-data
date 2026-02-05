"""Main simulation orchestrator."""

import re
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from pde import FieldCollection, FileStorage, MemoryStorage

from .config import COLORMAP_CYCLE, SimulationConfig, load_config
from .output import OutputManager, create_metadata
from ..boundaries import create_grid_with_bc
from ..pdes import get_pde_preset

# Valid backend options
VALID_BACKENDS = ("auto", "numpy", "numba")


def _get_next_folder_name(
    base_path: Path,
    preset: str,
    config_name: str | None = None,
    overwrite: bool = False,
    unique_suffix: bool = False,
) -> str:
    """Find the next available folder name for a preset.

    Args:
        base_path: Base output directory.
        preset: Name of the PDE preset.
        config_name: Name of the config file (without extension). If provided,
            folder names will be "{config_name}_{number}" (e.g., "directed_fast_001").
            If None, folder names will be just "{number}" (e.g., "001").
        overwrite: If True, return the last used number instead of incrementing.
        unique_suffix: If True, append a short random suffix to avoid collisions when
            running concurrently (e.g. Slurm array jobs).

    Returns:
        The next folder name (e.g., "directed_fast_001" or "001").
    """
    preset_dir = base_path / preset
    if not preset_dir.exists():
        number = "001"
        return f"{config_name}_{number}" if config_name else number

    # Build regex pattern based on whether we have a config name. We allow an
    # optional suffix so numbering still works for names like `foo_001_ab12cd`.
    if config_name:
        pattern = re.compile(rf"^{re.escape(config_name)}_(\d+)(?:_.+)?$")
    else:
        pattern = re.compile(r"^(\d+)(?:_.+)?$")

    # Find all existing numbered folders matching the pattern
    existing_numbers = []
    for item in preset_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing_numbers.append(int(match.group(1)))

    max_number = max(existing_numbers) if existing_numbers else 0

    if overwrite:
        number = f"{max_number:03d}" if max_number else "001"
        return f"{config_name}_{number}" if config_name else number

    number = f"{max_number + 1:03d}" if max_number else "001"
    base = f"{config_name}_{number}" if config_name else number
    if unique_suffix:
        # Short suffix: readable and collision-resistant enough for concurrent jobs.
        suffix = uuid.uuid4().hex[:8]
        return f"{base}_{suffix}"
    return base


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
        config_name: str | None = None,
        unique_suffix: bool | None = None,
    ):
        """Initialize the simulation runner.

        Args:
            config: Simulation configuration.
            output_dir: Override for output directory.
            sim_id: Override for simulation ID (auto-generated if None).
            overwrite: If True, overwrite the last numbered folder instead of creating a new one.
            config_name: Name of the config file (without extension). Used to prefix
                the output folder name (e.g., "directed_fast_001").
            unique_suffix: If True, append a short random suffix to avoid collisions.
                If None, uses `config.output.unique_suffix`.
        """
        self.config = config
        self.sim_id = sim_id or str(uuid.uuid4())
        self.unique_suffix = config.output.unique_suffix if unique_suffix is None else unique_suffix

        # Track whether output_dir was explicitly provided
        explicit_output_dir = output_dir is not None
        self.output_dir = Path(output_dir) if output_dir else config.output.path

        # Generate run name with incremental number (zero-padded, e.g., "001" or "directed_fast_001")
        # When explicit output_dir is provided, place numbered folders directly in that directory
        # When using default output path, add preset subdirectory
        if explicit_output_dir:
            # Look for existing runs directly in output_dir (no preset subdirectory)
            folder_name = _get_next_folder_name(
                self.output_dir,
                ".",
                config_name,
                overwrite,
                unique_suffix=self.unique_suffix and not overwrite,
            )
            self.run_name = folder_name
            self.folder_name = folder_name
        else:
            folder_name = _get_next_folder_name(
                self.output_dir,
                config.preset,
                config_name,
                overwrite,
                unique_suffix=self.unique_suffix and not overwrite,
            )
            self.run_name = folder_name
            # Full folder path: {preset}/{folder_name}
            self.folder_name = f"{config.preset}/{folder_name}"

        # Ensure the output directory exists early so file-based storage can write into it.
        self.full_output_dir = self.output_dir / self.folder_name
        self.full_output_dir.mkdir(parents=True, exist_ok=True)

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

        # Validate that this PDE supports the requested dimensionality
        self.preset.validate_dimension(config.ndim)

        # Create grid with appropriate boundary conditions
        self.grid = create_grid_with_bc(
            resolution=config.resolution,
            domain_size=config.domain_size,
            bc_config=config.bc,
        )

        # Create PDE
        params = config.parameters

        # Validate all required parameters are provided
        required_params = {p.name for p in self.preset.metadata.parameters}
        provided_params = set(params.keys())
        missing = required_params - provided_params
        if missing:
            raise ValueError(
                f"Missing required parameters for '{config.preset}': {sorted(missing)}"
            )

        # Warn about unused parameters in config
        unused_params = provided_params - required_params
        if unused_params:
            warnings.warn(
                f"Unused parameters in config for '{config.preset}': {sorted(unused_params)}. "
                f"Known parameters are: {sorted(required_params)}",
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
        # Inject global seed into IC params for reproducibility (if not already specified)
        ic_params = config.init.params.copy()
        if config.seed is not None and "seed" not in ic_params:
            ic_params["seed"] = config.seed
        ic_params = self.preset.resolve_ic_params(
            grid=self.grid,
            ic_type=config.init.type,
            ic_params=ic_params,
        )
        self.state = self.preset.create_initial_state(
            grid=self.grid,
            ic_type=config.init.type,
            ic_params=ic_params,
            parameters=params,
            bc=config.bc,
        )

        # Determine field configurations for output
        # Auto-assign colormaps from COLORMAP_CYCLE based on field order
        field_names = self.preset.metadata.field_names
        self.field_configs = [
            (name, COLORMAP_CYCLE[i % len(COLORMAP_CYCLE)])
            for i, name in enumerate(field_names)
        ]

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
            resolution_str = "x".join(str(r) for r in self.config.resolution)
            print(f"Starting simulation: {self.config.preset}")
            print(f"  Dimensions: {self.config.ndim}D")
            print(f"  Resolution: {resolution_str}")
            print(f"  Final time: {self.config.t_end}, dt: {self.config.dt}")
            print(f"  Frames: {num_frames} (interval: {save_interval:.4f})")
            print(f"  Solver: {self.config.solver}")
            print(f"  Backend: {self.config.backend}")
            # Only show adaptive info for solvers that support it
            if self._get_solver_name() != "implicit" and self.config.adaptive:
                print(f"  Adaptive: True (tolerance: {self.config.tolerance})")
            print(f"  Output: {self.full_output_dir}")
            print(f"  Formats: {', '.join(self.config.output.formats)}")

        def _extract_field_data(state: Any, field_name: str) -> np.ndarray:
            """Extract numpy data for a field name from a py-pde state."""
            if isinstance(state, FieldCollection):
                for field in state:
                    if hasattr(field, "label") and field.label == field_name:
                        data = field.data
                        break
                else:
                    # Fallback to first field
                    data = state[0].data
            else:
                data = state.data

            data = np.asarray(data)
            if np.iscomplexobj(data):
                data = np.abs(data)
            return data

        # Create storage for capturing frames. For large simulations, use FileStorage
        # to avoid holding all frames in RAM.
        storage_mode = (self.config.output.storage or "memory").lower()
        storage_path: Path | None = None
        if storage_mode == "file":
            storage_path = self.full_output_dir / "_py_pde_storage.h5"
            storage = FileStorage(storage_path, write_mode="truncate")
        elif storage_mode == "memory":
            storage = MemoryStorage()
        else:
            raise ValueError(
                f"Invalid output.storage '{self.config.output.storage}'. Use 'memory' or 'file'."
            )

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
        _result = self.pde.solve(self.state, **solve_kwargs)
        wall_clock_duration = time.perf_counter() - wall_clock_start

        # Capture solver diagnostics (including adaptive dt statistics)
        solver_diagnostics = getattr(self.pde, "diagnostics", {}).get("solver", {})

        # Save frames from storage (2-pass for consistent colorscale without loading
        # everything into memory).
        if verbose:
            print(f"  Saving {len(storage)} frames...")

        field_names = [name for name, _ in self.field_configs]
        field_ranges: dict[str, tuple[float, float]] = {
            name: (float("inf"), float("-inf")) for name in field_names
        }

        # Pass 1: compute global min/max per field
        for _t, state in storage.items():
            for name in field_names:
                data = _extract_field_data(state, name)
                vmin, vmax = field_ranges[name]
                field_ranges[name] = (min(vmin, float(np.min(data))), max(vmax, float(np.max(data))))

        # Initialize output manager after solve so handlers can allocate using the
        # actual number of frames written by py-pde.
        actual_num_frames = len(storage)
        self.output_manager = OutputManager(
            base_path=self.output_dir,
            folder_name=self.folder_name,
            colormap=COLORMAP_CYCLE[0],
            field_configs=self.field_configs,
            output_formats=self.config.output.formats,
            fps=self.config.output.fps,
            ndim=self.config.ndim,
            expected_num_frames=actual_num_frames,
        )
        self.output_manager.field_ranges = field_ranges

        # Pass 2: compute stagnation diagnostics and write outputs
        rel_threshold = 1e-4
        min_stagnant_fraction = 0.2

        max_abs_diff: dict[str, float] = {name: 0.0 for name in field_names}
        final_abs_diff: dict[str, float] = {name: 0.0 for name in field_names}
        trailing_stagnant: dict[str, int] = {name: 0 for name in field_names}
        prev_data: dict[str, np.ndarray] = {}

        for frame_index, (t, state) in enumerate(storage.items()):
            for name in field_names:
                data = _extract_field_data(state, name)
                vmin, vmax = field_ranges[name]
                field_range = vmax - vmin
                if field_range == 0:
                    continue
                if name in prev_data:
                    abs_diff = float(np.max(np.abs(data - prev_data[name])))
                    max_abs_diff[name] = max(max_abs_diff[name], abs_diff)
                    final_abs_diff[name] = abs_diff
                    if abs_diff < rel_threshold * field_range:
                        trailing_stagnant[name] += 1
                    else:
                        trailing_stagnant[name] = 0
                prev_data[name] = data.copy()

            self.output_manager.save_all_fields(state, frame_index, t)

        # Build stagnation report compatible with the old diagnostics structure
        stagnant_fields: list[str] = []
        fields_info: dict[str, dict[str, Any]] = {}
        min_stagnant_count = max(int(min_stagnant_fraction * actual_num_frames), 2)
        for name in field_names:
            vmin, vmax = field_ranges[name]
            field_range = vmax - vmin
            if field_range == 0:
                stagnant_fields.append(name)
                fields_info[name] = {
                    "stagnant": True,
                    "stagnant_from_frame": 0,
                    "trailing_stagnant_frames": actual_num_frames,
                    "field_range": 0.0,
                    "max_relative_change": 0.0,
                    "final_relative_change": 0.0,
                }
                continue

            trailing = trailing_stagnant[name]
            is_stagnant = trailing >= min_stagnant_count
            if is_stagnant:
                stagnant_fields.append(name)
            stagnant_from_frame = (actual_num_frames - 1 - trailing) if is_stagnant else None

            fields_info[name] = {
                "stagnant": is_stagnant,
                "stagnant_from_frame": stagnant_from_frame,
                "trailing_stagnant_frames": trailing,
                "field_range": field_range,
                "max_relative_change": max_abs_diff[name] / field_range,
                "final_relative_change": final_abs_diff[name] / field_range,
            }

        stagnation = {"stagnant_fields": stagnant_fields, "fields": fields_info}

        if verbose and stagnant_fields:
            for name in stagnant_fields:
                info = fields_info[name]
                if info["field_range"] == 0:
                    print(f"  WARNING: Field '{name}' is completely constant")
                else:
                    print(
                        f"  WARNING: Field '{name}' appears stagnant from frame {info['stagnant_from_frame']} "
                        f"({info['trailing_stagnant_frames']} frames with no significant change)"
                    )

        # Generate and save metadata
        total_time = storage.times[-1] if len(storage) > 0 else self.config.t_end

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

        # Add stagnation diagnostics to metadata
        metadata["diagnostics"] = {"stagnation": stagnation}

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

        # Clean up file-based intermediate storage unless requested.
        if storage_path is not None:
            try:
                storage.close()
            except Exception:
                pass
            if not self.config.output.keep_storage:
                try:
                    storage_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return metadata


def run_from_config(
    config_path: Path | str,
    output_dir: Path | str | None = None,
    seed: int | None = None,
    verbose: bool = True,
    overwrite: bool = False,
    storage: str | None = None,
    keep_storage: bool | None = None,
    unique_suffix: bool | None = None,
) -> dict[str, Any]:
    """Run a simulation from a config file.

    Args:
        config_path: Path to YAML configuration file.
        output_dir: Override for output directory.
        seed: Override for random seed.
        verbose: Whether to print progress.
        overwrite: If True, overwrite the last numbered folder instead of creating a new one.
        storage: Override for output.storage ("memory" or "file").
        keep_storage: Override for output.keep_storage.
        unique_suffix: Override for output.unique_suffix.

    Returns:
        Simulation metadata dictionary.
    """
    config_path = Path(config_path)
    config = load_config(config_path)

    # Extract config name from filename (without .yaml extension)
    config_name = config_path.stem

    # Override seed if provided
    if seed is not None:
        config.seed = seed
    if storage is not None:
        config.output.storage = storage
    if keep_storage is not None:
        config.output.keep_storage = keep_storage
    if unique_suffix is not None:
        config.output.unique_suffix = unique_suffix

    runner = SimulationRunner(
        config,
        output_dir=output_dir,
        overwrite=overwrite,
        config_name=config_name,
        unique_suffix=unique_suffix,
    )
    return runner.run(verbose=verbose)
