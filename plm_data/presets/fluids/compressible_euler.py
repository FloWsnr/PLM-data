"""2D compressible Euler preset using a Cartesian finite-volume scheme."""

import math

import numpy as np
from dolfinx import default_real_type, fem, mesh
from mpi4py import MPI

from plm_data.core.initial_conditions import (
    build_scalar_ic_interpolator,
    build_vector_ic_interpolator,
)
from plm_data.core.logging import get_logger
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

_COMPRESSIBLE_EULER_SPEC = PresetSpec(
    name="compressible_euler",
    category="fluids",
    description=(
        "Two-dimensional compressible Euler equations in conservative form on a "
        "Cartesian finite-volume grid with local Lax-Friedrichs fluxes."
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
        PDEParameter("cfl", "Maximum CFL number for adaptive explicit stepping."),
        PDEParameter(
            "density_floor",
            "Minimum density enforced after each timestep for admissibility.",
        ),
        PDEParameter(
            "pressure_floor",
            "Minimum pressure enforced after each timestep for admissibility.",
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

    def _setup_runtime(self) -> None:
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
        self.comm = MPI.COMM_WORLD
        if self.comm.size > self.ny:
            raise ValueError(
                f"Preset '{self.spec.name}' requires at least one y-row per MPI rank. "
                f"Got {self.comm.size} ranks for ny={self.ny}."
            )
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.max_dt = float(self.config.time.dt)

        state_boundary_field = self.config.boundary_field("state")
        self._boundary_types = {
            side: state_boundary_field.side_conditions(side)[0].type
            for side in ("x-", "x+", "y-", "y+")
        }

        self._setup_parallel_layout()
        self._build_output_mesh()
        self._initialize_state()
        self._latest_metrics = self._state_metrics(correction_count=0)

    def _setup_parallel_layout(self) -> None:
        base_rows = self.ny // self.comm.size
        extra_rows = self.ny % self.comm.size
        self._row_counts = np.array(
            [
                base_rows + (1 if rank < extra_rows else 0)
                for rank in range(self.comm.size)
            ],
            dtype=np.int32,
        )
        self._row_offsets = np.zeros(self.comm.size, dtype=np.int32)
        if self.comm.size > 1:
            self._row_offsets[1:] = np.cumsum(self._row_counts[:-1])

        self._local_row_start = int(self._row_offsets[self.comm.rank])
        self._local_ny = int(self._row_counts[self.comm.rank])
        self._local_row_stop = self._local_row_start + self._local_ny

        self._lower_rank = (
            self.comm.rank - 1
            if self.comm.rank > 0
            else (
                self.comm.size - 1
                if self._boundary_types["y-"] == "periodic" and self.comm.size > 1
                else MPI.PROC_NULL
            )
        )
        self._upper_rank = (
            self.comm.rank + 1
            if self.comm.rank < self.comm.size - 1
            else (
                0
                if self._boundary_types["y+"] == "periodic" and self.comm.size > 1
                else MPI.PROC_NULL
            )
        )
        self._gather_counts = [int(count) * self.nx * 4 for count in self._row_counts]
        self._gather_displs = [0]
        for count in self._gather_counts[:-1]:
            self._gather_displs.append(self._gather_displs[-1] + count)

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
        self._num_dofs = 4 * self.nx * self.ny

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

    def _initialize_state(self) -> None:
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

        x_centers = (np.arange(self.nx, dtype=float) + 0.5) * self.dx
        y_centers = (
            np.arange(self._local_row_start, self._local_row_stop, dtype=float) + 0.5
        ) * self.dy
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
        points = np.zeros((3, self._local_ny * self.nx), dtype=float)
        points[0, :] = xx.ravel()
        points[1, :] = yy.ravel()

        density = np.asarray(density_interp(points), dtype=float).reshape(
            self._local_ny,
            self.nx,
        )
        pressure = np.asarray(pressure_interp(points), dtype=float).reshape(
            self._local_ny,
            self.nx,
        )
        velocity = np.asarray(velocity_interp(points), dtype=float)
        if velocity.shape != (2, self._local_ny * self.nx):
            raise ValueError(
                f"Velocity initial condition produced shape {velocity.shape}, expected "
                f"(2, {self._local_ny * self.nx})."
            )
        velocity_x = velocity[0, :].reshape(self._local_ny, self.nx)
        velocity_y = velocity[1, :].reshape(self._local_ny, self.nx)

        local_min_density = float(np.min(density)) if density.size > 0 else float("inf")
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

        self._state_local = self._primitive_to_conservative(
            density,
            velocity_x,
            velocity_y,
            pressure,
        )

    def _primitive_to_conservative(
        self,
        density: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        momentum_x = density * velocity_x
        momentum_y = density * velocity_y
        total_energy = pressure / (self.gamma - 1.0) + 0.5 * density * (
            velocity_x**2 + velocity_y**2
        )
        return np.stack((density, momentum_x, momentum_y, total_energy), axis=-1)

    def _primitive_fields(
        self,
        state: np.ndarray,
        *,
        clip_pressure: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        density = np.maximum(state[..., 0], self.density_floor)
        velocity_x = state[..., 1] / density
        velocity_y = state[..., 2] / density
        kinetic = 0.5 * density * (velocity_x**2 + velocity_y**2)
        pressure = (self.gamma - 1.0) * (state[..., 3] - kinetic)
        if clip_pressure:
            pressure = np.maximum(pressure, self.pressure_floor)
        return density, velocity_x, velocity_y, pressure

    def _flux_x(self, state: np.ndarray) -> np.ndarray:
        _, velocity_x, _, pressure = self._primitive_fields(state, clip_pressure=True)
        total_energy = state[..., 3]
        return np.stack(
            (
                state[..., 1],
                state[..., 1] * velocity_x + pressure,
                state[..., 2] * velocity_x,
                (total_energy + pressure) * velocity_x,
            ),
            axis=-1,
        )

    def _flux_y(self, state: np.ndarray) -> np.ndarray:
        _, _, velocity_y, pressure = self._primitive_fields(state, clip_pressure=True)
        total_energy = state[..., 3]
        return np.stack(
            (
                state[..., 2],
                state[..., 1] * velocity_y,
                state[..., 2] * velocity_y + pressure,
                (total_energy + pressure) * velocity_y,
            ),
            axis=-1,
        )

    def _rusanov_flux(
        self,
        left_state: np.ndarray,
        right_state: np.ndarray,
        *,
        axis: int,
    ) -> np.ndarray:
        if axis == 0:
            left_flux = self._flux_x(left_state)
            right_flux = self._flux_x(right_state)
        else:
            left_flux = self._flux_y(left_state)
            right_flux = self._flux_y(right_state)

        left_density, left_u, left_v, left_pressure = self._primitive_fields(
            left_state,
            clip_pressure=True,
        )
        right_density, right_u, right_v, right_pressure = self._primitive_fields(
            right_state,
            clip_pressure=True,
        )
        left_sound = np.sqrt(self.gamma * left_pressure / left_density)
        right_sound = np.sqrt(self.gamma * right_pressure / right_density)
        if axis == 0:
            left_speed = np.abs(left_u) + left_sound
            right_speed = np.abs(right_u) + right_sound
        else:
            left_speed = np.abs(left_v) + left_sound
            right_speed = np.abs(right_v) + right_sound

        wave_speed = np.maximum(left_speed, right_speed)[..., None]
        return 0.5 * (left_flux + right_flux) - 0.5 * wave_speed * (
            right_state - left_state
        )

    def _reflect_x(self, state: np.ndarray) -> np.ndarray:
        reflected = np.array(state, copy=True)
        reflected[..., 1] *= -1.0
        return reflected

    def _reflect_y(self, state: np.ndarray) -> np.ndarray:
        reflected = np.array(state, copy=True)
        reflected[..., 2] *= -1.0
        return reflected

    def _x_minus_ghost(self, state: np.ndarray) -> np.ndarray:
        if self._boundary_types["x-"] == "periodic":
            return np.array(state[:, -1, :], copy=True)
        return self._reflect_x(state[:, 0, :])

    def _x_plus_ghost(self, state: np.ndarray) -> np.ndarray:
        if self._boundary_types["x+"] == "periodic":
            return np.array(state[:, 0, :], copy=True)
        return self._reflect_x(state[:, -1, :])

    def _y_halos(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._local_ny == 0:
            raise RuntimeError("Cannot exchange halos for an empty local state.")

        if self.comm.size == 1:
            bottom_halo = (
                np.array(state[-1, :, :], copy=True)
                if self._boundary_types["y-"] == "periodic"
                else self._reflect_y(state[0, :, :])
            )
            top_halo = (
                np.array(state[0, :, :], copy=True)
                if self._boundary_types["y+"] == "periodic"
                else self._reflect_y(state[-1, :, :])
            )
            return bottom_halo, top_halo

        bottom_halo = np.empty((self.nx, 4), dtype=state.dtype)
        top_halo = np.empty((self.nx, 4), dtype=state.dtype)
        requests: list[MPI.Request] = []
        send_buffers: list[np.ndarray] = []

        if self._lower_rank == MPI.PROC_NULL:
            bottom_halo[:] = self._reflect_y(state[0, :, :])
        else:
            requests.append(
                self.comm.Irecv(bottom_halo, source=self._lower_rank, tag=22)
            )

        if self._upper_rank == MPI.PROC_NULL:
            top_halo[:] = self._reflect_y(state[-1, :, :])
        else:
            requests.append(self.comm.Irecv(top_halo, source=self._upper_rank, tag=11))

        if self._lower_rank != MPI.PROC_NULL:
            send_to_lower = np.ascontiguousarray(state[0, :, :])
            send_buffers.append(send_to_lower)
            requests.append(
                self.comm.Isend(send_to_lower, dest=self._lower_rank, tag=11)
            )

        if self._upper_rank != MPI.PROC_NULL:
            send_to_upper = np.ascontiguousarray(state[-1, :, :])
            send_buffers.append(send_to_upper)
            requests.append(
                self.comm.Isend(send_to_upper, dest=self._upper_rank, tag=22)
            )

        if requests:
            MPI.Request.Waitall(requests)

        return bottom_halo, top_halo

    def _compute_stable_dt(self) -> float:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            self._state_local,
            clip_pressure=True,
        )
        if density.size == 0:
            local_max_inverse_dt = 0.0
        else:
            sound_speed = np.sqrt(self.gamma * pressure / density)
            inverse_dt = (np.abs(velocity_x) + sound_speed) / self.dx + (
                np.abs(velocity_y) + sound_speed
            ) / self.dy
            local_max_inverse_dt = float(np.max(inverse_dt))

        max_inverse_dt = self.comm.allreduce(local_max_inverse_dt, op=MPI.MAX)
        if not math.isfinite(max_inverse_dt):
            raise RuntimeError(
                f"Preset '{self.spec.name}' produced a non-finite wave speed."
            )
        if max_inverse_dt <= 0.0:
            return self.max_dt
        return self.cfl / max_inverse_dt

    def _apply_admissibility_floors(self, state: np.ndarray) -> int:
        if not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"Preset '{self.spec.name}' produced non-finite conservative state values."
            )

        density = state[..., 0]
        density_corrections = density < self.density_floor
        if np.any(density_corrections):
            density[density_corrections] = self.density_floor

        momentum_sq = state[..., 1] ** 2 + state[..., 2] ** 2
        min_total_energy = self.pressure_floor / (self.gamma - 1.0) + 0.5 * (
            momentum_sq / density
        )
        total_energy = state[..., 3]
        energy_corrections = total_energy < min_total_energy
        if np.any(energy_corrections):
            total_energy[energy_corrections] = min_total_energy[energy_corrections]

        return int(np.count_nonzero(density_corrections | energy_corrections))

    def _advance_local_state(self, dt: float) -> int:
        state = self._state_local
        bottom_halo, top_halo = self._y_halos(state)
        state_x = np.concatenate(
            (
                self._x_minus_ghost(state)[:, None, :],
                state,
                self._x_plus_ghost(state)[:, None, :],
            ),
            axis=1,
        )
        state_y = np.concatenate(
            (
                bottom_halo[None, :, :],
                state,
                top_halo[None, :, :],
            ),
            axis=0,
        )
        flux_x = self._rusanov_flux(state_x[:, :-1, :], state_x[:, 1:, :], axis=0)
        flux_y = self._rusanov_flux(state_y[:-1, :, :], state_y[1:, :, :], axis=1)

        updated_state = (
            state
            - (dt / self.dx) * (flux_x[:, 1:, :] - flux_x[:, :-1, :])
            - (dt / self.dy) * (flux_y[1:, :, :] - flux_y[:-1, :, :])
        )
        correction_count = self._apply_admissibility_floors(updated_state)
        self._state_local = updated_state
        return correction_count

    def _state_metrics(
        self,
        *,
        correction_count: int,
    ) -> dict[str, float | int]:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            self._state_local,
            clip_pressure=False,
        )
        if density.size == 0:
            local_min_density = float("inf")
            local_min_pressure = float("inf")
            local_max_wave_speed = 0.0
        else:
            sound_speed = np.sqrt(
                self.gamma * np.maximum(pressure, self.pressure_floor) / density
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
        if min_density <= 0.0:
            raise ValueError(
                f"Preset '{self.spec.name}' produced non-positive density. "
                f"Minimum density: {min_density:.6g}."
            )
        if min_pressure <= 0.0:
            raise ValueError(
                f"Preset '{self.spec.name}' produced non-positive pressure. "
                f"Minimum pressure: {min_pressure:.6g}."
            )
        return {
            "min_density": min_density,
            "min_pressure": min_pressure,
            "max_wave_speed": max_wave_speed,
            "admissibility_corrections": total_corrections,
        }

    def _gather_global_state(self) -> np.ndarray:
        recv = np.empty(self.nx * self.ny * 4, dtype=self._state_local.dtype)
        self.comm.Allgatherv(
            np.ascontiguousarray(self._state_local).reshape(-1),
            (recv, self._gather_counts, self._gather_displs, MPI.DOUBLE),
        )
        return recv.reshape(self.ny, self.nx, 4)

    def _update_output_fields(self, state: np.ndarray) -> None:
        density, velocity_x, velocity_y, pressure = self._primitive_fields(
            state,
            clip_pressure=False,
        )
        local_density = density[self._owned_cell_j, self._owned_cell_i]
        local_pressure = pressure[self._owned_cell_j, self._owned_cell_i]
        local_energy = state[self._owned_cell_j, self._owned_cell_i, 3]
        local_velocity = np.stack(
            (
                velocity_x[self._owned_cell_j, self._owned_cell_i],
                velocity_y[self._owned_cell_j, self._owned_cell_i],
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

    def _write_current_frame(self, output, *, t: float) -> None:
        state = self._gather_global_state()
        self._update_output_fields(state)
        output.write_frame(
            {
                "density": self.density_out,
                "velocity": self.velocity_out,
                "pressure": self.pressure_out,
                "total_energy": self.total_energy_out,
            },
            t=t,
        )

    def run(self, output) -> RunResult:
        logger = get_logger("timestepper")
        self._setup_runtime()

        assert self.config.time is not None
        dt_limit = self.max_dt
        t_end = float(self.config.time.t_end)
        num_frames = self.config.output.num_frames
        if num_frames > 1:
            output_times = np.linspace(0.0, t_end, num_frames)
        else:
            output_times = np.array([t_end], dtype=float)
        next_output_idx = 0

        estimated_steps = max(1, int(math.ceil(t_end / max(dt_limit, 1.0e-12))))
        log_every = max(1, estimated_steps // 10)
        logger.info(
            "  Time stepping: adaptive explicit finite volume, dt_max=%s, t_end=%s",
            dt_limit,
            t_end,
        )

        self.record_runtime_health("initialization")
        if next_output_idx < len(output_times):
            self._write_current_frame(output, t=0.0)
            next_output_idx += 1

        t = 0.0
        num_steps = 0
        while t < t_end - 1.0e-14 * max(1.0, t_end):
            stable_dt = self._compute_stable_dt()
            step_dt = min(dt_limit, stable_dt, t_end - t)
            if next_output_idx < len(output_times):
                remaining_to_output = float(output_times[next_output_idx] - t)
                if remaining_to_output > 1.0e-14 * max(1.0, dt_limit):
                    step_dt = min(step_dt, remaining_to_output)
            if step_dt <= 0.0:
                raise RuntimeError(
                    f"Preset '{self.spec.name}' selected a non-positive timestep "
                    f"{step_dt} at t={t:.6g}."
                )

            correction_count = self._advance_local_state(step_dt)
            self._latest_metrics = self._state_metrics(
                correction_count=correction_count
            )

            t += step_dt
            num_steps += 1
            self.record_solver_health(f"step_{num_steps}")
            self.record_runtime_health(f"t={t:.6g}")

            if num_steps % log_every == 0 or num_steps == 1:
                progress = t / t_end * 100.0 if t_end > 0.0 else 100.0
                logger.info(
                    "  Step %d (t=%.4g, %.0f%%)",
                    num_steps,
                    t,
                    progress,
                )

            if next_output_idx < len(output_times) and t >= output_times[
                next_output_idx
            ] - 1.0e-14 * max(step_dt, 1.0):
                self._write_current_frame(output, t=t)
                next_output_idx += 1

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
