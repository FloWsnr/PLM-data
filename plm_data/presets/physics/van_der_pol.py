"""Diffusively coupled Van der Pol oscillator preset."""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
)

_VAN_DER_POL_SPEC = PresetSpec(
    name="van_der_pol",
    category="physics",
    description=(
        "Diffusively coupled Van der Pol oscillators. Two-field reaction-diffusion "
        "system with displacement u and velocity v."
    ),
    equations={
        "u": "du/dt = Du * laplacian(u) + v",
        "v": "dv/dt = Dv * laplacian(v) + mu * (1 - u^2) * v - u",
    },
    parameters=[
        PDEParameter("Du", "Diffusion coefficient of the displacement field u"),
        PDEParameter("Dv", "Diffusion coefficient of the velocity field v"),
        PDEParameter(
            "mu", "Nonlinear damping coefficient controlling oscillation character"
        ),
    ],
    inputs={
        "u": InputSpec(
            name="u",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
        "v": InputSpec(
            name="v",
            shape="scalar",
            allow_source=False,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "u": BoundaryFieldSpec(
            name="u",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the displacement field u.",
        ),
        "v": BoundaryFieldSpec(
            name="v",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field v.",
        ),
    },
    states={
        "u": StateSpec(name="u", shape="scalar"),
        "v": StateSpec(name="v", shape="scalar"),
    },
    outputs={
        "u": OutputSpec(
            name="u",
            shape="scalar",
            output_mode="scalar",
            source_name="u",
        ),
        "v": OutputSpec(
            name="v",
            shape="scalar",
            output_mode="scalar",
            source_name="v",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _global_axis_bounds(msh) -> tuple[tuple[float, float], ...]:
    coords = msh.geometry.x
    comm = msh.comm
    bounds: list[tuple[float, float]] = []
    for axis in range(msh.geometry.dim):
        local_min = float(coords[:, axis].min()) if coords.size > 0 else np.inf
        local_max = float(coords[:, axis].max()) if coords.size > 0 else -np.inf
        bounds.append(
            (
                comm.allreduce(local_min, op=MPI.MIN),
                comm.allreduce(local_max, op=MPI.MAX),
            )
        )
    return tuple(bounds)


def _periodic_random_modes_interpolator(
    *,
    amplitude: float,
    num_modes: int,
    max_wavenumber: int,
    seed: int | None,
    bounds: tuple[tuple[float, float], ...],
):
    if num_modes <= 0:
        raise ValueError("periodic_random_modes requires num_modes > 0")
    if max_wavenumber <= 0:
        raise ValueError("periodic_random_modes requires max_wavenumber > 0")

    rng = np.random.default_rng(seed)
    gdim = len(bounds)
    offsets = np.array([lower for lower, _ in bounds], dtype=float)
    lengths = np.array([upper - lower for lower, upper in bounds], dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("periodic_random_modes requires positive domain lengths")

    wave_numbers = rng.integers(1, max_wavenumber + 1, size=(num_modes, gdim))
    trig_selector = rng.integers(0, 2, size=(num_modes, gdim))
    coefficients = rng.normal(size=num_modes)
    coefficients /= max(float(np.linalg.norm(coefficients)), 1.0)
    angular_frequencies = 2.0 * np.pi * wave_numbers / lengths

    def _interpolator(x: np.ndarray) -> np.ndarray:
        shifted = x[:gdim, :] - offsets[:, None]
        values = np.zeros(x.shape[1], dtype=float)
        for mode in range(num_modes):
            mode_values = np.ones(x.shape[1], dtype=float)
            for axis in range(gdim):
                argument = angular_frequencies[mode, axis] * shifted[axis]
                trig_fn = np.cos if trig_selector[mode, axis] else np.sin
                mode_values *= trig_fn(argument)
            values += coefficients[mode] * mode_values
        return amplitude * values

    return _interpolator


def _apply_van_der_pol_ic(
    func: fem.Function,
    *,
    ic_config,
    parameters: dict[str, float],
    seed: int | None,
    bounds: tuple[tuple[float, float], ...],
) -> None:
    if ic_config.type == "periodic_random_modes":
        amplitude = float(ic_config.params["amplitude"])
        num_modes = int(ic_config.params["num_modes"])
        max_wavenumber = int(ic_config.params["max_wavenumber"])
        func.interpolate(
            _periodic_random_modes_interpolator(
                amplitude=amplitude,
                num_modes=num_modes,
                max_wavenumber=max_wavenumber,
                seed=seed,
                bounds=bounds,
            )
        )
        return

    apply_ic(
        func,
        ic_config,
        parameters,
        seed=seed,
    )


class _VanDerPolProblem(TransientLinearProblem):
    supported_solver_strategies = (CONSTANT_LHS_SCALAR_SPD,)

    def validate_boundary_conditions(self, domain_geom):
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="u",
            boundary_field=self.config.boundary_field("u"),
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="v",
            boundary_field=self.config.boundary_field("v"),
            domain_geom=domain_geom,
        )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.msh = domain_geom.mesh
        bounds = _global_axis_bounds(self.msh)
        V = fem.functionspace(self.msh, ("Lagrange", 1))

        dt = self.config.time.dt
        params = self.config.parameters
        u_boundary_field = self.config.boundary_field("u")
        v_boundary_field = self.config.boundary_field("v")

        u_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            u_boundary_field,
            params,
        )
        v_bcs = apply_dirichlet_bcs(
            V,
            domain_geom,
            v_boundary_field,
            params,
        )
        u_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            u_boundary_field,
            u_bcs,
        )
        v_mpc = self.create_periodic_constraint(
            V,
            domain_geom,
            v_boundary_field,
            v_bcs,
        )

        self.V_u = V if u_mpc is None else u_mpc.function_space
        self.V_v = V if v_mpc is None else v_mpc.function_space
        self._num_dofs = _space_num_dofs(self.V_u) + _space_num_dofs(self.V_v)

        self.u_n = fem.Function(self.V_u, name="u")
        self.v_n = fem.Function(self.V_v, name="v")
        self.u_h = fem.Function(self.V_u, name="u_next")
        self.v_h = fem.Function(self.V_v, name="v_next")

        u_initial_condition = self.config.input("u").initial_condition
        v_initial_condition = self.config.input("v").initial_condition
        assert u_initial_condition is not None
        assert v_initial_condition is not None
        _apply_van_der_pol_ic(
            self.u_n,
            ic_config=u_initial_condition,
            parameters=params,
            seed=self.config.seed,
            bounds=bounds,
        )
        self.u_n.x.scatter_forward()
        if u_mpc is not None:
            u_mpc.backsubstitution(self.u_n)
        _apply_van_der_pol_ic(
            self.v_n,
            ic_config=v_initial_condition,
            parameters=params,
            seed=(self.config.seed + 1) if self.config.seed is not None else None,
            bounds=bounds,
        )
        self.v_n.x.scatter_forward()
        if v_mpc is not None:
            v_mpc.backsubstitution(self.v_n)

        u_trial = ufl.TrialFunction(V)
        u_test = ufl.TestFunction(V)
        v_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)

        a_u = (
            ufl.inner(u_trial, u_test) * ufl.dx
            + dt
            * params["Du"]
            * ufl.inner(ufl.grad(u_trial), ufl.grad(u_test))
            * ufl.dx
        )
        L_u = (
            ufl.inner(self.u_n, u_test) * ufl.dx
            + dt * ufl.inner(self.v_n, u_test) * ufl.dx
        )

        v_rhs = params["mu"] * (1.0 - self.u_n**2) * self.v_n - self.u_n
        a_v = (
            ufl.inner(v_trial, v_test) * ufl.dx
            + dt
            * params["Dv"]
            * ufl.inner(ufl.grad(v_trial), ufl.grad(v_test))
            * ufl.dx
        )
        L_v = (
            ufl.inner(self.v_n, v_test) * ufl.dx
            + dt * ufl.inner(v_rhs, v_test) * ufl.dx
        )

        a_u_bc, L_u_bc = build_natural_bc_forms(
            u_trial,
            u_test,
            domain_geom,
            u_boundary_field,
            params,
        )
        if a_u_bc is not None:
            a_u = a_u + dt * a_u_bc
        if L_u_bc is not None:
            L_u = L_u + dt * L_u_bc

        a_v_bc, L_v_bc = build_natural_bc_forms(
            v_trial,
            v_test,
            domain_geom,
            v_boundary_field,
            params,
        )
        if a_v_bc is not None:
            a_v = a_v + dt * a_v_bc
        if L_v_bc is not None:
            L_v = L_v + dt * L_v_bc

        self._u_problem = self.create_linear_problem(
            a_u,
            L_u,
            u=self.u_h,
            bcs=u_bcs,
            petsc_options_prefix="plm_van_der_pol_u_",
            mpc=u_mpc,
        )
        self._v_problem = self.create_linear_problem(
            a_v,
            L_v,
            u=self.v_h,
            bcs=v_bcs,
            petsc_options_prefix="plm_van_der_pol_v_",
            mpc=v_mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._u_problem.solve()
        self._v_problem.solve()

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.v_n.x.array[:] = self.v_h.x.array
        self.v_n.x.scatter_forward()

        return (
            self._u_problem.solver.getConvergedReason() > 0
            and self._v_problem.solver.getConvergedReason() > 0
        )

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {"u": self.u_n, "v": self.v_n}

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("van_der_pol")
class VanDerPolPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _VAN_DER_POL_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _VanDerPolProblem(self.spec, config)
