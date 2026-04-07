"""2D compressible Euler preset using Clawpack's classic solver."""

import importlib
from pathlib import Path
import time

import numpy as np
from dolfinx import default_real_type, fem, mesh
from mpi4py import MPI

from plm_data.core.initial_conditions import (
    build_scalar_ic_interpolator,
    build_vector_ic_interpolator,
)
from plm_data.core.logging import configure_clawpack_logging, get_logger
from plm_data.core.solver_strategies import TRANSIENT_EXPLICIT
from plm_data.presets import register_preset
from plm_data.presets.base import CustomProblem, PDEPreset, ProblemInstance, RunResult
from plm_data.presets.boundary_validation import validate_boundary_field_structure
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    BoundaryOperatorSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_REFLECTIVE_BOUNDARY_OPERATOR = BoundaryOperatorSpec(
    name="reflective",
    value_shape=None,
    description=(
        "Reflective slip-wall boundary for the full Euler state. "
        "Normal momentum reverses sign; density and total energy reflect unchanged."
    ),
)

_EULER_STATE_BOUNDARY_OPERATORS = {
    "periodic": SCALAR_STANDARD_BOUNDARY_OPERATORS["periodic"],
    "reflective": _REFLECTIVE_BOUNDARY_OPERATOR,
}

# Clawpack's 2D Euler conservative-variable ordering.
DENSITY = 0
X_MOMENTUM = 1
Y_MOMENTUM = 2
ENERGY = 3
NUM_EQN = 4

_COMPRESSIBLE_EULER_SPEC = PresetSpec(
    name="compressible_euler",
    category="fluids",
    description=(
        "Two-dimensional compressible Euler equations in conservative form using "
        "the mandatory Clawpack HLLE wave-propagation solver with admissibility "
        "repair."
    ),
    equations={
        "density": "d(rho)/dt + div(momentum) = 0",
        "momentum": ("d(momentum)/dt + div(momentum ⊗ velocity + pressure * I) = 0"),
        "total_energy": (
            "d(E)/dt + div((E + pressure) * velocity) = 0, "
            "pressure = (gamma - 1) * (E - 0.5 * rho * |velocity|^2)"
        ),
    },
    parameters=[
        PDEParameter("gamma", "Ratio of specific heats."),
        PDEParameter("cfl", "Desired CFL number for the Clawpack solver."),
        PDEParameter(
            "density_floor",
            "Minimum density used for derived-field clipping and validation.",
        ),
        PDEParameter(
            "pressure_floor",
            "Minimum pressure used for derived-field clipping and validation.",
        ),
    ],
    inputs={
        "density": InputSpec(
            name="density",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "pressure": InputSpec(
            name="pressure",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "state": BoundaryFieldSpec(
            name="state",
            shape="scalar",
            operators=_EULER_STATE_BOUNDARY_OPERATORS,
            description=(
                "Boundary operator for the full conservative Euler state. "
                "Supports periodic pairs and reflective slip walls."
            ),
        )
    },
    states={
        "density": StateSpec(name="density", shape="scalar"),
        "momentum": StateSpec(name="momentum", shape="vector"),
        "total_energy": StateSpec(name="total_energy", shape="scalar"),
    },
    outputs={
        "density": OutputSpec(
            name="density",
            shape="scalar",
            output_mode="scalar",
            source_name="density",
        ),
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
            source_kind="derived",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
            source_kind="derived",
        ),
        "total_energy": OutputSpec(
            name="total_energy",
            shape="scalar",
            output_mode="scalar",
            source_name="total_energy",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2],
)


def _positive_parameter(parameters: dict[str, float], name: str) -> float:
    value = float(parameters[name])
    if value <= 0.0:
        raise ValueError(
            f"Preset 'compressible_euler' requires parameter '{name}' > 0. Got {value}."
        )
    return value


class _CompressibleEulerProblem(CustomProblem):
    supported_solver_strategies = (TRANSIENT_EXPLICIT,)

    def validate_boundary_conditions(self, domain_geom) -> None:
        if domain_geom.mesh.geometry.dim != 2:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D domains, got "
                f"{domain_geom.mesh.geometry.dim}D."
            )

        expected_boundary_names = {"x-", "x+", "y-", "y+"}
        actual_boundary_names = set(domain_geom.boundary_names)
        if actual_boundary_names != expected_boundary_names:
            raise ValueError(
                f"Preset '{self.spec.name}' requires a rectangle domain with "
                f"boundary names {sorted(expected_boundary_names)}. Got "
                f"{sorted(actual_boundary_names)}."
            )

        validate_boundary_field_structure(
            preset_name=self.spec.name,
            field_name="state",
            boundary_field=self.config.boundary_field("state"),
            domain_geom=domain_geom,
            allowed_operators={"periodic", "reflective"},
        )

    def runtime_health_metrics(self, context: str) -> dict[str, float | int]:
        return getattr(self, "_latest_metrics", {})

    def _load_clawpack_backend(self) -> None:
        try:
            self._pyclaw = importlib.import_module("clawpack.pyclaw")
            self._petclaw = importlib.import_module("clawpack.petclaw")
            self._riemann = importlib.import_module("clawpack.riemann")
        except ImportError as exc:
            raise RuntimeError(
                f"Preset '{self.spec.name}' requires the mandatory 'clawpack' "
                "dependency in the active environment."
            ) from exc

        configure_clawpack_logging()
        Path("pyclaw.log").unlink(missing_ok=True)

    def _install_progress_logging(self) -> None:
        self._progress_logger = get_logger("timestepper")
        self._progress_wall_start = time.monotonic()
        self._last_progress_log_wall_time = self._progress_wall_start

        def _progress_before_step(solver, state) -> None:
            if self.comm.rank != 0:
                return
            now = time.monotonic()
            if now - self._last_progress_log_wall_time < 30.0:
                return
            self._progress_logger.info(
                "  Progress: step %d, t=%.6g, dt=%.3g, wall=%.0fs",
                int(solver.status["numsteps"]),
                float(self._solution.t),
                float(solver.dt),
                now - self._progress_wall_start,
            )
            self._last_progress_log_wall_time = now

        self._solver.before_step = _progress_before_step

    def _setup_runtime(self) -> None:
        logger = get_logger("timestepper")
        if self.config.time is None:
            raise ValueError(f"Preset '{self.spec.name}' requires a time section.")
        if self.config.domain.type != "rectangle":
            raise ValueError(
                f"Preset '{self.spec.name}' only supports the rectangle domain. "
                f"Got '{self.config.domain.type}'."
            )
        if self.config.domain.periodic_maps:
            raise ValueError(
                f"Preset '{self.spec.name}' does not accept custom domain.periodic_maps."
            )

        self.load_domain_geometry()

        params = self.config.parameters
        self.gamma = _positive_parameter(params, "gamma")
        self.cfl = _positive_parameter(params, "cfl")
        if self.cfl >= 1.0:
            raise ValueError(
                f"Preset '{self.spec.name}' requires parameter 'cfl' < 1. "
                f"Got {self.cfl}."
            )
        self.density_floor = _positive_parameter(params, "density_floor")
        self.pressure_floor = _positive_parameter(params, "pressure_floor")

        size = self.config.domain.params["size"]
        resolution = self.config.domain.params["mesh_resolution"]
        self.Lx = float(size[0])
        self.Ly = float(size[1])
        self.nx = int(resolution[0])
        self.ny = int(resolution[1])
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.max_dt = float(self.config.time.dt)
        self.comm = MPI.COMM_WORLD

        state_boundary_field = self.config.boundary_field("state")
        self._boundary_types = {
            side: state_boundary_field.side_conditions(side)[0].type
            for side in ("x-", "x+", "y-", "y+")
        }

        self._load_clawpack_backend()
        logger.info("  Setup: output sampling")
        self._setup_output_sampling()
        logger.info("  Setup: output mesh")
        self._build_output_mesh()
        logger.info("  Setup: Clawpack state")
        self._initialize_clawpack_solution()
        self._install_progress_logging()
        logger.info("  Setup: runtime health")
        self._latest_metrics = self._state_metrics(self._local_q(), correction_count=0)

    def _build_output_mesh(self) -> None:
        self.output_mesh = mesh.create_rectangle(
            MPI.COMM_WORLD,
            ((0.0, 0.0), (self.Lx, self.Ly)),
            (self.nx, self.ny),
            cell_type=mesh.CellType.quadrilateral,
            ghost_mode=mesh.GhostMode.shared_facet,
        )

        scalar_space = fem.functionspace(self.output_mesh, ("DG", 0))
        vector_space = fem.functionspace(
            self.output_mesh,
            ("Discontinuous Lagrange", 0, (2,)),
        )
        self.density_out = fem.Function(scalar_space, name="density")
        self.pressure_out = fem.Function(scalar_space, name="pressure")
        self.total_energy_out = fem.Function(scalar_space, name="total_energy")
        self.velocity_out = fem.Function(vector_space, name="velocity")

        tdim = self.output_mesh.topology.dim
        cell_map = self.output_mesh.topology.index_map(tdim)
        owned_cell_ids = np.arange(cell_map.size_local, dtype=np.int32)
        self._num_dofs = NUM_EQN * self.nx * self.ny

        cell_midpoints = mesh.compute_midpoints(self.output_mesh, tdim, owned_cell_ids)
        self._owned_cell_i = np.clip(
            np.floor(cell_midpoints[:, 0] / self.dx).astype(np.int32),
            0,
            self.nx - 1,
        )
        self._owned_cell_j = np.clip(
            np.floor(cell_midpoints[:, 1] / self.dy).astype(np.int32),
            0,
            self.ny - 1,
        )
        self._owned_scalar_cell_blocks = np.array(
            [scalar_space.dofmap.cell_dofs(cell)[0] for cell in owned_cell_ids],
            dtype=np.int32,
        )
        self._owned_vector_cell_blocks = np.array(
            [vector_space.dofmap.cell_dofs(cell)[0] for cell in owned_cell_ids],
            dtype=np.int32,
        )

    def _setup_output_sampling(self) -> None:
        output_nx, output_ny = self.config.output.resolution
        x_coords = np.linspace(0.0, self.Lx, output_nx)
        y_coords = np.linspace(0.0, self.Ly, output_ny)
        self._output_x_indices = np.clip(
            np.floor(x_coords / self.dx).astype(np.int32),
            0,
            self.nx - 1,
        )
        self._output_y_indices = np.clip(
            np.floor(y_coords / self.dy).astype(np.int32),
            0,
            self.ny - 1,
        )

    def _configure_boundary_conditions(self) -> None:
        bc_map = {
            "periodic": self._claw.BC.periodic,
            "reflective": self._claw.BC.wall,
        }
        self._solver.bc_lower[0] = bc_map[self._boundary_types["x-"]]
        self._solver.bc_upper[0] = bc_map[self._boundary_types["x+"]]
        self._solver.bc_lower[1] = bc_map[self._boundary_types["y-"]]
        self._solver.bc_upper[1] = bc_map[self._boundary_types["y+"]]

    def _initialize_clawpack_solution(self) -> None:
        self._claw = self._petclaw if self.comm.size > 1 else self._pyclaw
        self._last_correction_count = 0

        density_input = self.config.input("density")
        velocity_input = self.config.input("velocity")
        pressure_input = self.config.input("pressure")
        assert density_input.initial_condition is not None
        assert velocity_input.initial_condition is not None
        assert pressure_input.initial_condition is not None

        density_interp = build_scalar_ic_interpolator(
            density_input.initial_condition,
            self.config.parameters,
            gdim=2,
            seed=self.config.seed,
            stream_id="density",
        )
        velocity_interp = build_vector_ic_interpolator(
            velocity_input.initial_condition,
            self.config.parameters,
            gdim=2,
            seed=self.config.seed,
            stream_id="velocity",
        )
        pressure_interp = build_scalar_ic_interpolator(
            pressure_input.initial_condition,
            self.config.parameters,
            gdim=2,
            seed=self.config.seed,
            stream_id="pressure",
        )
        if density_interp is None or velocity_interp is None or pressure_interp is None:
            raise ValueError(
                f"Preset '{self.spec.name}' does not support custom initial "
                "conditions for density, velocity, or pressure."
            )

        x = self._claw.Dimension(0.0, self.Lx, self.nx, name="x")
        y = self._claw.Dimension(0.0, self.Ly, self.ny, name="y")
        domain = self._claw.Domain([x, y])
        state = self._claw.State(domain, NUM_EQN)
        state.problem_data["gamma"] = self.gamma
        state.problem_data["gamma1"] = self.gamma - 1.0

        centers = state.grid.p_centers
        if centers is None or len(centers) != 2:
            raise RuntimeError(
                f"Preset '{self.spec.name}' could not access Clawpack cell centers."
            )
        xx, yy = centers
        points = np.zeros((3, xx.size), dtype=float)
        points[0, :] = xx.ravel()
        points[1, :] = yy.ravel()

        density = np.asarray(density_interp(points), dtype=float).reshape(xx.shape)
        pressure = np.asarray(pressure_interp(points), dtype=float).reshape(xx.shape)
        velocity = np.asarray(velocity_interp(points), dtype=float)
        if velocity.shape != (2, xx.size):
            raise ValueError(
                f"Velocity initial condition produced shape {velocity.shape}, expected "
                f"(2, {xx.size})."
            )
        velocity_x = velocity[0, :].reshape(xx.shape)
        velocity_y = velocity[1, :].reshape(xx.shape)

        local_min_density = (
            float(np.min(density)) if density.size > 0 else float("inf")
        )
        local_min_pressure = (
            float(np.min(pressure)) if pressure.size > 0 else float("inf")
        )
        min_density = self.comm.allreduce(local_min_density, op=MPI.MIN)
        min_pressure = self.comm.allreduce(local_min_pressure, op=MPI.MIN)
        if min_density <= self.density_floor:
            raise ValueError(
                f"Preset '{self.spec.name}' initial density must stay above "
                f"density_floor={self.density_floor:.6g}. Got min {min_density:.6g}."
            )
        if min_pressure <= self.pressure_floor:
            raise ValueError(
                f"Preset '{self.spec.name}' initial pressure must stay above "
                f"pressure_floor={self.pressure_floor:.6g}. Got min {min_pressure:.6g}."
            )

        q = state.q
        if q is None:
            raise RuntimeError(
                f"Preset '{self.spec.name}' could not access the Clawpack state array."
            )
        q[DENSITY, ...] = density
        q[X_MOMENTUM, ...] = density * velocity_x
        q[Y_MOMENTUM, ...] = density * velocity_y
        q[ENERGY, ...] = pressure / (self.gamma - 1.0) + 0.5 * density * (
            velocity_x**2 + velocity_y**2
        )

        self._solution = self._claw.Solution(state, domain)
        self._solver = self._claw.ClawSolver2D(self._riemann.euler_hlle_2D)
        self._solver.transverse_waves = 0
        self._solver.limiters = self._pyclaw.limiters.tvd.MC
        self._solver.cfl_desired = self.cfl
        self._solver.cfl_max = min(0.5, max(self.cfl + 0.05, self.cfl * 1.1))
        self._solver.dt_initial = min(self.max_dt, self._compute_stable_dt(state.q))
        self._solver.dt_max = self.max_dt
        self._solver.max_steps = 1_000_000
        self._configure_boundary_conditions()
        self._install_admissibility_repair()
        self._solver.setup(self._solution)
        self._solver.dt = self._solver.dt_initial

    def _install_admissibility_repair(self) -> None:
        original_step = self._solver.step

        def _patched_step(solution, take_one_step, tstart, tend):
            accepted = original_step(solution, take_one_step, tstart, tend)
            q = solution.state.q
            if q is None:
                raise RuntimeError(
                    f"Preset '{self.spec.name}' could not access the Clawpack state "
                    "array after a timestep."
                )

            density = q[DENSITY, ...]
            density_bad = (~np.isfinite(density)) | (density < self.density_floor)
            if np.any(density_bad):
                density[density_bad] = self.density_floor

            momentum_x = q[X_MOMENTUM, ...]
            momentum_y = q[Y_MOMENTUM, ...]
            bad_momentum_x = ~np.isfinite(momentum_x)
            bad_momentum_y = ~np.isfinite(momentum_y)
            if np.any(bad_momentum_x):
                momentum_x[bad_momentum_x] = 0.0
            if np.any(bad_momentum_y):
                momentum_y[bad_momentum_y] = 0.0

            density_safe = np.maximum(density, self.density_floor)
            momentum_sq = momentum_x**2 + momentum_y**2
            min_total_energy = self.pressure_floor / (self.gamma - 1.0) + 0.5 * (
                momentum_sq / density_safe
            )
            total_energy = q[ENERGY, ...]
            energy_bad = (~np.isfinite(total_energy)) | (
                total_energy < min_total_energy
            )
            if np.any(energy_bad):
                total_energy[energy_bad] = min_total_energy[energy_bad]

            self._last_correction_count = int(
                np.count_nonzero(
                    density_bad | bad_momentum_x | bad_momentum_y | energy_bad
                )
            )
            return accepted

        self._solver.step = _patched_step

    def _local_q(self) -> np.ndarray:
        return np.asarray(self._solution.state.q, dtype=float)

    def _global_q(self) -> np.ndarray:
        if self.comm.size == 1:
            global_q = np.array(self._solution.state.q, copy=True)
        else:
            global_q = self._solution.state.get_q_global()
        global_q = self.comm.bcast(global_q, root=0)
        if global_q is None:
            raise RuntimeError(
                f"Preset '{self.spec.name}' could not gather the global Clawpack state."
            )
        return np.asarray(global_q, dtype=float)

    def _primitive_fields(
        self,
        q: np.ndarray,
        *,
        clip_pressure: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        density = np.asarray(q[DENSITY, ...], dtype=float)
        safe_density = np.maximum(density, self.density_floor)
        momentum_x = np.asarray(q[X_MOMENTUM, ...], dtype=float)
        momentum_y = np.asarray(q[Y_MOMENTUM, ...], dtype=float)
        total_energy = np.asarray(q[ENERGY, ...], dtype=float)
        velocity_x = momentum_x / safe_density
        velocity_y = momentum_y / safe_density
        kinetic = 0.5 * (momentum_x**2 + momentum_y**2) / safe_density
        pressure = (self.gamma - 1.0) * (total_energy - kinetic)
        if clip_pressure:
            pressure = np.maximum(pressure, self.pressure_floor)
        return density, velocity_x, velocity_y, pressure

    def _compute_stable_dt(self, q: np.ndarray) -> float:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            q,
            clip_pressure=True,
        )
        if density.size == 0:
            local_max_inverse_dt = 0.0
        else:
            sound_speed = np.sqrt(self.gamma * pressure / np.maximum(density, self.density_floor))
            inverse_dt = (np.abs(velocity_x) + sound_speed) / self.dx + (
                np.abs(velocity_y) + sound_speed
            ) / self.dy
            local_max_inverse_dt = float(np.max(inverse_dt))

        max_inverse_dt = self.comm.allreduce(local_max_inverse_dt, op=MPI.MAX)
        if not np.isfinite(max_inverse_dt):
            raise RuntimeError(
                f"Preset '{self.spec.name}' produced a non-finite wave speed."
            )
        if max_inverse_dt <= 0.0:
            return self.max_dt
        return self.cfl / max_inverse_dt

    def _state_metrics(
        self,
        q: np.ndarray,
        *,
        correction_count: int,
    ) -> dict[str, float | int]:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            q,
            clip_pressure=False,
        )
        if density.size == 0:
            local_min_density = float("inf")
            local_min_pressure = float("inf")
            local_max_wave_speed = 0.0
        else:
            safe_density = np.maximum(density, self.density_floor)
            sound_speed = np.sqrt(
                self.gamma * np.maximum(pressure, self.pressure_floor) / safe_density
            )
            local_max_wave_speed = float(
                np.max(
                    np.maximum(
                        np.abs(velocity_x) + sound_speed,
                        np.abs(velocity_y) + sound_speed,
                    )
                )
            )
            local_min_density = float(np.min(density))
            local_min_pressure = float(np.min(pressure))

        max_wave_speed = self.comm.allreduce(local_max_wave_speed, op=MPI.MAX)
        min_density = self.comm.allreduce(local_min_density, op=MPI.MIN)
        min_pressure = self.comm.allreduce(local_min_pressure, op=MPI.MIN)
        total_corrections = self.comm.allreduce(int(correction_count), op=MPI.SUM)
        if not np.isfinite(min_density) or min_density <= 0.0:
            raise RuntimeError(
                f"Preset '{self.spec.name}' produced non-positive density. "
                f"Minimum density: {min_density:.6g}."
            )
        if not np.isfinite(min_pressure) or min_pressure <= 0.0:
            raise RuntimeError(
                f"Preset '{self.spec.name}' produced non-positive pressure. "
                f"Minimum pressure: {min_pressure:.6g}."
            )
        return {
            "min_density": min_density,
            "min_pressure": min_pressure,
            "max_wave_speed": max_wave_speed,
            "admissibility_corrections": total_corrections,
        }

    def _update_output_fields(self, q: np.ndarray) -> None:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            q,
            clip_pressure=False,
        )
        local_density = density[self._owned_cell_i, self._owned_cell_j]
        local_pressure = pressure[self._owned_cell_i, self._owned_cell_j]
        local_energy = q[ENERGY, self._owned_cell_i, self._owned_cell_j]
        local_velocity = np.stack(
            (
                velocity_x[self._owned_cell_i, self._owned_cell_j],
                velocity_y[self._owned_cell_i, self._owned_cell_j],
            ),
            axis=1,
        )

        self.density_out.x.array[self._owned_scalar_cell_blocks] = np.asarray(
            local_density,
            dtype=default_real_type,
        )
        self.pressure_out.x.array[self._owned_scalar_cell_blocks] = np.asarray(
            local_pressure,
            dtype=default_real_type,
        )
        self.total_energy_out.x.array[self._owned_scalar_cell_blocks] = np.asarray(
            local_energy,
            dtype=default_real_type,
        )
        velocity_array = self.velocity_out.x.array.reshape(-1, 2)
        velocity_array[self._owned_vector_cell_blocks, :] = np.asarray(
            local_velocity,
            dtype=default_real_type,
        )

        self.density_out.x.scatter_forward()
        self.pressure_out.x.scatter_forward()
        self.total_energy_out.x.scatter_forward()
        self.velocity_out.x.scatter_forward()

    def _sample_output_fields(self, q: np.ndarray) -> dict[str, np.ndarray]:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            q,
            clip_pressure=False,
        )
        return {
            "density": density[
                self._output_x_indices[:, None],
                self._output_y_indices[None, :],
            ],
            "velocity_x": velocity_x[
                self._output_x_indices[:, None],
                self._output_y_indices[None, :],
            ],
            "velocity_y": velocity_y[
                self._output_x_indices[:, None],
                self._output_y_indices[None, :],
            ],
            "pressure": pressure[
                self._output_x_indices[:, None],
                self._output_y_indices[None, :],
            ],
            "total_energy": q[
                ENERGY,
                self._output_x_indices[:, None],
                self._output_y_indices[None, :],
            ],
        }

    def _write_current_frame(self, output, *, t: float) -> None:
        q = self._global_q()
        sampled_fields = self._sample_output_fields(q)
        fem_fields: dict[str, fem.Function] = {}
        if output.needs_fem_fields():
            self._update_output_fields(q)
            fem_fields = {
                "density": self.density_out,
                "velocity": self.velocity_out,
                "pressure": self.pressure_out,
                "total_energy": self.total_energy_out,
            }
        output.write_frame(fem_fields, t=t, sampled_fields=sampled_fields)

    def run(self, output) -> RunResult:
        logger = get_logger("timestepper")
        self._setup_runtime()

        assert self.config.time is not None
        t_end = float(self.config.time.t_end)
        num_frames = self.config.output.num_frames
        if num_frames > 1:
            output_times = np.linspace(0.0, t_end, num_frames)
        else:
            output_times = np.array([t_end], dtype=float)
        next_output_idx = 0
        next_log_progress = 0.1

        logger.info(
            "  Time stepping: Clawpack wave propagation, dt_max=%s, t_end=%s",
            self.max_dt,
            t_end,
        )

        self.record_runtime_health("initialization")
        if next_output_idx < len(output_times):
            self._write_current_frame(output, t=0.0)
            next_output_idx += 1

        while self._solution.t < t_end - 1.0e-14 * max(1.0, t_end):
            target_time = t_end
            if next_output_idx < len(output_times):
                target_time = min(target_time, float(output_times[next_output_idx]))

            status = self._solver.evolve_to_time(self._solution, target_time)
            current_t = float(self._solution.t)
            if current_t < target_time - 1.0e-12 * max(1.0, target_time):
                raise RuntimeError(
                    f"Preset '{self.spec.name}' did not reach target time "
                    f"{target_time:.6g}; stopped at {current_t:.6g}."
                )

            self._latest_metrics = self._state_metrics(
                self._local_q(),
                correction_count=self._last_correction_count,
            )
            self.record_runtime_health(f"t={current_t:.6g}")

            progress = current_t / t_end if t_end > 0.0 else 1.0
            if progress >= next_log_progress - 1.0e-12:
                logger.info(
                    "  Step %d (t=%.4g, %.0f%%)",
                    int(status["numsteps"]),
                    current_t,
                    progress * 100.0,
                )
                while progress >= next_log_progress - 1.0e-12:
                    next_log_progress += 0.1

            if next_output_idx < len(output_times) and current_t >= output_times[
                next_output_idx
            ] - 1.0e-12 * max(1.0, target_time):
                self._write_current_frame(output, t=current_t)
                next_output_idx += 1

        num_steps = int(self._solver.status["numsteps"])
        logger.info("  Time stepping complete: %d substeps", num_steps)
        return RunResult(
            num_dofs=self._num_dofs,
            solver_converged=True,
            num_timesteps=num_steps,
            diagnostics=self.build_health_diagnostics(),
        )


@register_preset("compressible_euler")
class CompressibleEulerPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _COMPRESSIBLE_EULER_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _CompressibleEulerProblem(self.spec, config)
