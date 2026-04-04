"""Shared stochastic forcing and random-media helpers."""

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.config import (
    CoefficientSmoothingConfig,
    CoefficientStochasticConfig,
    FieldExpressionConfig,
    StateStochasticConfig,
)
from plm_data.core.spatial_fields import (
    build_interpolator,
    build_ufl_field,
    scalar_expression_to_config,
)

if TYPE_CHECKING:
    from plm_data.presets.base import ProblemInstance

_SPLITMIX_CONST_0 = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_CONST_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_CONST_2 = np.uint64(0x94D049BB133111EB)
_MASK_53 = np.uint64((1 << 53) - 1)


def _reference_cell_point(tdim: int) -> np.ndarray:
    return np.full((1, tdim), 1.0 / (tdim + 1.0), dtype=float)


def _stream_seed(seed: int | None, stream_id: str, comm) -> np.uint64:
    if seed is None:
        raise ValueError(
            "Stochastic sampling requires an explicit seed from the config or '--seed'."
        )

    digest = hashlib.blake2b(
        f"{seed}:{stream_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return np.uint64(int.from_bytes(digest, "little"))


def _splitmix64(values: np.ndarray) -> np.ndarray:
    z = values.astype(np.uint64, copy=False) + _SPLITMIX_CONST_0
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_CONST_1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_CONST_2
    return z ^ (z >> np.uint64(31))


def _hashed_standard_normal(cell_ids: np.ndarray, stream_seed: np.uint64) -> np.ndarray:
    keys = cell_ids.astype(np.uint64, copy=False) ^ stream_seed
    u1_bits = _splitmix64(keys)
    u2_bits = _splitmix64(keys + _SPLITMIX_CONST_0)
    u1 = ((u1_bits & _MASK_53).astype(np.float64) + 0.5) / float(1 << 53)
    u2 = ((u2_bits & _MASK_53).astype(np.float64) + 0.5) / float(1 << 53)
    u1 = np.clip(u1, np.finfo(np.float64).tiny, 1.0)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def _global_cell_ids(mesh) -> np.ndarray:
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    local_indices = np.arange(
        cell_map.size_local + cell_map.num_ghosts,
        dtype=np.int32,
    )
    return cell_map.local_to_global(local_indices)


def _cell_volumes(mesh) -> np.ndarray:
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    expr = fem.Expression(
        ufl.CellVolume(mesh),
        _reference_cell_point(mesh.topology.dim),
        dtype=default_real_type,
    )
    values = expr.eval(mesh, cells)
    return np.asarray(values, dtype=float).reshape(num_cells, -1)[:, 0]


@dataclass
class _ScalarCellNoise:
    mesh: object
    seed: int | None
    stream_root: str
    volume_scaling: bool

    def __post_init__(self) -> None:
        self.space = fem.functionspace(self.mesh, ("DG", 0))
        self.function = fem.Function(self.space, name=self.stream_root)
        self._cell_ids = _global_cell_ids(self.mesh)
        num_cells = self._cell_ids.size
        if num_cells != self.function.x.array.size:
            raise ValueError(
                f"DG0 noise sampler expected {num_cells} cell dofs, got "
                f"{self.function.x.array.size}."
            )
        self._scale = np.ones(num_cells, dtype=float)
        if self.volume_scaling:
            volumes = _cell_volumes(self.mesh)
            if volumes.shape[0] != num_cells:
                raise ValueError(
                    f"Cell-volume sampler expected {num_cells} cells, got "
                    f"{volumes.shape[0]}."
                )
            if np.any(volumes <= 0.0):
                raise ValueError(
                    "Cell volumes must be positive for stochastic scaling."
                )
            self._scale = 1.0 / np.sqrt(volumes)

    def fill(self, step_index: int | None = None) -> None:
        stream_id = self.stream_root
        if step_index is not None:
            stream_id = f"{stream_id}:{step_index}"
        seed = _stream_seed(self.seed, stream_id, self.mesh.comm)
        values = _hashed_standard_normal(self._cell_ids, seed) * self._scale
        self.function.x.array[:] = values.astype(
            self.function.x.array.dtype, copy=False
        )
        self.function.x.scatter_forward()


class DynamicStateNoiseRuntime:
    """Per-timestep stochastic forcing for one state variable."""

    def __init__(
        self,
        mesh,
        *,
        seed: int | None,
        stream_root: str,
        dt: float,
        state_shape: str,
        stochastic: StateStochasticConfig,
    ) -> None:
        self.mesh = mesh
        self.state_shape = state_shape
        self.stochastic = stochastic
        self._scale = fem.Constant(
            mesh,
            default_real_type(np.sqrt(dt) * stochastic.intensity),
        )
        if state_shape == "scalar":
            component_names = ("scalar",)
        elif state_shape == "vector":
            component_names = tuple(f"component_{i}" for i in range(mesh.geometry.dim))
        else:
            raise ValueError(f"Unsupported stochastic state shape '{state_shape}'.")

        self._components = [
            _ScalarCellNoise(
                mesh,
                seed=seed,
                stream_root=f"{stream_root}.{component_name}",
                volume_scaling=True,
            )
            for component_name in component_names
        ]

    def update(self, step_index: int) -> None:
        for component in self._components:
            component.fill(step_index)

    def forcing_expr(self, previous_state):
        coupling = self.stochastic.coupling
        if self.state_shape == "scalar":
            noise_expr = self._components[0].function
            if coupling == "additive":
                base_expr = noise_expr
            elif coupling == "multiplicative_self":
                base_expr = previous_state * noise_expr
            elif coupling == "saturating_self":
                assert self.stochastic.offset is not None
                base_expr = (
                    previous_state / (previous_state + self.stochastic.offset)
                ) * noise_expr
            else:
                raise ValueError(
                    f"Unsupported scalar stochastic coupling '{coupling}'."
                )
            return self._scale * base_expr

        if coupling == "saturating_self":
            raise ValueError(
                "Vector stochastic states do not support 'saturating_self' in v1."
            )

        component_exprs = []
        for index, component in enumerate(self._components):
            noise_expr = component.function
            if coupling == "additive":
                component_exprs.append(self._scale * noise_expr)
            elif coupling == "multiplicative_self":
                component_exprs.append(self._scale * previous_state[index] * noise_expr)
            else:
                raise ValueError(
                    f"Unsupported vector stochastic coupling '{coupling}'."
                )
        return ufl.as_vector(component_exprs)


def build_scalar_state_stochastic_term(
    problem: "ProblemInstance",
    *,
    state_name: str,
    previous_state,
    test,
    dt: float,
) -> tuple[ufl.Form | None, DynamicStateNoiseRuntime | None]:
    """Build a scalar stochastic forcing contribution for one state."""
    stochastic = problem.config.stochastic_state(state_name)
    if stochastic is None or stochastic.intensity == 0.0:
        return None, None

    runtime = DynamicStateNoiseRuntime(
        problem.msh,
        seed=problem.config.seed,
        stream_root=f"{problem.spec.name}.state.{state_name}",
        dt=dt,
        state_shape="scalar",
        stochastic=stochastic,
    )
    return ufl.inner(runtime.forcing_expr(previous_state), test) * ufl.dx, runtime


def build_vector_state_stochastic_term(
    problem: "ProblemInstance",
    *,
    state_name: str,
    previous_state,
    test,
    dt: float,
) -> tuple[ufl.Form | None, DynamicStateNoiseRuntime | None]:
    """Build a vector stochastic forcing contribution for one state."""
    stochastic = problem.config.stochastic_state(state_name)
    if stochastic is None or stochastic.intensity == 0.0:
        return None, None

    runtime = DynamicStateNoiseRuntime(
        problem.msh,
        seed=problem.config.seed,
        stream_root=f"{problem.spec.name}.state.{state_name}",
        dt=dt,
        state_shape="vector",
        stochastic=stochastic,
    )
    return ufl.inner(runtime.forcing_expr(previous_state), test) * ufl.dx, runtime


def _interpolate_scalar_expression(
    function: fem.Function,
    expression: FieldExpressionConfig,
    parameters: dict[str, float],
    *,
    context: str,
) -> None:
    interpolator = build_interpolator(
        scalar_expression_to_config(expression),
        parameters,
    )
    if interpolator is None:
        raise ValueError(f"{context} cannot use a custom scalar expression.")
    function.interpolate(interpolator)
    function.x.scatter_forward()


def _solve_linear_problem(problem, *, context: str) -> None:
    problem.solve()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"{context} did not converge (KSP reason={reason})")


def _project_scalar_noise(
    problem: "ProblemInstance",
    source: fem.Function,
    *,
    name: str,
) -> fem.Function:
    target_space = fem.functionspace(problem.msh, ("Lagrange", 1))
    target = fem.Function(target_space, name=name)
    trial = ufl.TrialFunction(target_space)
    test = ufl.TestFunction(target_space)
    projection_problem = problem.create_linear_problem(
        ufl.inner(trial, test) * ufl.dx,
        ufl.inner(source, test) * ufl.dx,
        u=target,
        bcs=[],
        petsc_options_prefix=f"plm_stochastic_{problem.spec.name}_{name}_project_",
    )
    _solve_linear_problem(
        projection_problem,
        context=f"Stochastic projection '{name}'",
    )
    return target


def _diffusion_smooth(
    problem: "ProblemInstance",
    field: fem.Function,
    *,
    name: str,
    smoothing: CoefficientSmoothingConfig,
) -> fem.Function:
    solution = fem.Function(field.function_space, name=name)
    previous = fem.Function(field.function_space, name=f"{name}_prev")
    solution.x.array[:] = field.x.array
    solution.x.scatter_forward()

    trial = ufl.TrialFunction(field.function_space)
    test = ufl.TestFunction(field.function_space)
    tau = fem.Constant(problem.msh, default_real_type(smoothing.pseudo_dt))
    smoothing_problem = problem.create_linear_problem(
        ufl.inner(trial, test) * ufl.dx
        + tau * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx,
        ufl.inner(previous, test) * ufl.dx,
        u=solution,
        bcs=[],
        petsc_options_prefix=f"plm_stochastic_{problem.spec.name}_{name}_smooth_",
    )

    for _ in range(smoothing.steps):
        previous.x.array[:] = solution.x.array
        previous.x.scatter_forward()
        _solve_linear_problem(
            smoothing_problem,
            context=f"Stochastic smoothing '{name}'",
        )

    return solution


def _randomized_scalar_coefficient(
    problem: "ProblemInstance",
    *,
    name: str,
    base_expression: FieldExpressionConfig,
    stochastic: CoefficientStochasticConfig,
) -> fem.Function:
    stream_root = f"{problem.spec.name}.coefficient.{name}"
    overlay_sampler = _ScalarCellNoise(
        problem.msh,
        seed=problem.config.seed,
        stream_root=stream_root,
        volume_scaling=False,
    )
    overlay_sampler.fill()

    if stochastic.smoothing is None:
        coefficient_space = overlay_sampler.space
        overlay_function = overlay_sampler.function
    else:
        overlay_function = _project_scalar_noise(
            problem,
            overlay_sampler.function,
            name=f"{name}_overlay",
        )
        overlay_function = _diffusion_smooth(
            problem,
            overlay_function,
            name=f"{name}_overlay",
            smoothing=stochastic.smoothing,
        )
        coefficient_space = overlay_function.function_space

    coefficient = fem.Function(coefficient_space, name=name)
    _interpolate_scalar_expression(
        coefficient,
        base_expression,
        problem.config.parameters,
        context=f"Coefficient '{name}'",
    )

    overlay = overlay_function.x.array
    if stochastic.mode == "additive":
        coefficient.x.array[:] = coefficient.x.array + stochastic.std * overlay
    elif stochastic.mode == "multiplicative":
        coefficient.x.array[:] = coefficient.x.array * (1.0 + stochastic.std * overlay)
    else:
        raise ValueError(
            f"Unsupported stochastic coefficient mode '{stochastic.mode}'."
        )

    if stochastic.clamp_min is not None:
        coefficient.x.array[:] = np.maximum(coefficient.x.array, stochastic.clamp_min)
    coefficient.x.scatter_forward()
    return coefficient


def build_scalar_coefficient(
    problem: "ProblemInstance",
    name: str,
):
    """Build a scalar coefficient, optionally randomized as a static medium."""
    coefficient_expression = problem.config.coefficient(name)
    stochastic = problem.config.stochastic_coefficient(name)
    if stochastic is not None and stochastic.std > 0.0:
        return _randomized_scalar_coefficient(
            problem,
            name=name,
            base_expression=coefficient_expression,
            stochastic=stochastic,
        )

    return build_ufl_field(
        problem.msh,
        scalar_expression_to_config(coefficient_expression),
        problem.config.parameters,
    )
