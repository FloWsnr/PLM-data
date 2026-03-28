"""Thermal convection (Rayleigh-Benard) preset."""

import numpy as np
import ufl
from dolfinx import default_real_type, fem

from plm_data.core.boundary_conditions import (
    apply_dirichlet_bcs,
    build_natural_bc_forms,
    build_vector_natural_bc_forms,
)
from plm_data.core.initial_conditions import apply_ic, apply_vector_ic
from plm_data.core.source_terms import build_source_form, build_vector_source_form
from plm_data.presets import register_preset
from plm_data.presets.base import PDEPreset, ProblemInstance, TransientLinearProblem
from plm_data.presets.boundary_validation import (
    validate_scalar_standard_boundary_field,
    validate_vector_standard_boundary_field,
)
from plm_data.presets.fluids._taylor_hood import (
    create_taylor_hood_system,
    normalize_pressure,
)
from plm_data.presets.metadata import (
    BoundaryFieldSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PresetSpec,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    StateSpec,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_THERMAL_CONVECTION_SPEC = PresetSpec(
    name="thermal_convection",
    category="fluids",
    description=(
        "Thermal convection using the Boussinesq Rayleigh-Benard equations "
        "with a Taylor-Hood velocity/pressure discretization and a scalar "
        "temperature field."
    ),
    equations={
        "velocity": (
            "du/dt + (u_prev.grad)u = -grad(p) + sqrt(Pr/Ra)*laplacian(u) + T*e_v"
        ),
        "pressure": "div(u) = 0",
        "temperature": ("dT/dt + u_prev.grad(T) = (1/sqrt(Ra*Pr))*laplacian(T) + f_T"),
    },
    parameters=[
        PDEParameter("Ra", "Rayleigh number"),
        PDEParameter("Pr", "Prandtl number"),
        PDEParameter("k", "Pressure space degree; velocity/temperature use k+1"),
    ],
    inputs={
        "velocity": InputSpec(
            name="velocity",
            shape="vector",
            allow_source=True,
            allow_initial_condition=True,
        ),
        "temperature": InputSpec(
            name="temperature",
            shape="scalar",
            allow_source=True,
            allow_initial_condition=True,
        ),
    },
    boundary_fields={
        "velocity": BoundaryFieldSpec(
            name="velocity",
            shape="vector",
            operators=VECTOR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the velocity field.",
        ),
        "temperature": BoundaryFieldSpec(
            name="temperature",
            shape="scalar",
            operators=SCALAR_STANDARD_BOUNDARY_OPERATORS,
            description="Boundary conditions for the temperature field.",
        ),
    },
    states={
        "velocity": StateSpec(name="velocity", shape="vector"),
        "pressure": StateSpec(name="pressure", shape="scalar"),
        "temperature": StateSpec(name="temperature", shape="scalar"),
    },
    outputs={
        "velocity": OutputSpec(
            name="velocity",
            shape="vector",
            output_mode="components",
            source_name="velocity",
        ),
        "pressure": OutputSpec(
            name="pressure",
            shape="scalar",
            output_mode="scalar",
            source_name="pressure",
        ),
        "temperature": OutputSpec(
            name="temperature",
            shape="scalar",
            output_mode="scalar",
            source_name="temperature",
        ),
    },
    static_fields=[],
    steady_state=False,
    supported_dimensions=[2, 3],
)


def _expected_boundary_names(gdim: int) -> set[str]:
    if gdim == 2:
        return {"x-", "x+", "y-", "y+"}
    return {"x-", "x+", "y-", "y+", "z-", "z+"}


def _vertical_axis(gdim: int) -> int:
    if gdim == 2:
        return 1
    return 2


def _space_num_dofs(V: fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_global * V.dofmap.index_map_bs


def _constant_dirichlet_value(
    *,
    boundary_field,
    side_name: str,
    parameters: dict[str, float],
) -> float:
    entries = boundary_field.side_conditions(side_name)
    if len(entries) != 1 or entries[0].type != "dirichlet":
        raise ValueError(
            "conductive_noise initial conditions require constant Dirichlet "
            f"temperature values on '{side_name}'."
        )

    entry = entries[0]
    assert entry.value is not None
    if entry.value.is_componentwise:
        raise ValueError(
            f"Temperature boundary '{side_name}' must be scalar for conductive_noise."
        )
    if entry.value.type != "constant":
        raise ValueError(
            "conductive_noise initial conditions require constant Dirichlet "
            f"temperature values on '{side_name}'."
        )

    raw_value = entry.value.params["value"]
    if isinstance(raw_value, str) and raw_value.startswith("param:"):
        parameter_name = raw_value.split(":", maxsplit=1)[1]
        return parameters[parameter_name]
    return float(raw_value)


def _conductive_noise_interpolator(
    *,
    domain_geom,
    temperature_boundary_field,
    parameters: dict[str, float],
    amplitude: float,
    num_modes: int,
    seed: int | None,
):
    gdim = domain_geom.mesh.geometry.dim
    vertical_axis = _vertical_axis(gdim)
    vertical_minus = "y-" if gdim == 2 else "z-"
    vertical_plus = "y+" if gdim == 2 else "z+"

    bottom_temperature = _constant_dirichlet_value(
        boundary_field=temperature_boundary_field,
        side_name=vertical_minus,
        parameters=parameters,
    )
    top_temperature = _constant_dirichlet_value(
        boundary_field=temperature_boundary_field,
        side_name=vertical_plus,
        parameters=parameters,
    )

    coordinates = domain_geom.mesh.geometry.x
    lower_bounds = coordinates.min(axis=0)
    upper_bounds = coordinates.max(axis=0)
    spans = upper_bounds - lower_bounds
    horizontal_axes = [axis for axis in range(gdim) if axis != vertical_axis]

    rng = np.random.default_rng(seed)
    if gdim == 2:
        coefficients = rng.normal(size=num_modes)
        coefficient_scale = max(1.0, np.sqrt(float(num_modes)))
    else:
        coefficients = rng.normal(size=(num_modes, num_modes))
        coefficient_scale = max(1.0, np.sqrt(float(num_modes * num_modes)))

    def _interpolator(x: np.ndarray) -> np.ndarray:
        eta = (x[vertical_axis] - lower_bounds[vertical_axis]) / spans[vertical_axis]
        base_profile = bottom_temperature + (top_temperature - bottom_temperature) * eta
        vertical_envelope = np.sin(np.pi * eta)

        perturbation = np.zeros(x.shape[1], dtype=float)
        if gdim == 2:
            xi = (x[horizontal_axes[0]] - lower_bounds[horizontal_axes[0]]) / spans[
                horizontal_axes[0]
            ]
            for mode_index in range(num_modes):
                wave_number = mode_index + 1
                perturbation += coefficients[mode_index] * np.sin(
                    2.0 * np.pi * wave_number * xi
                )
        else:
            xi = (x[horizontal_axes[0]] - lower_bounds[horizontal_axes[0]]) / spans[
                horizontal_axes[0]
            ]
            zeta = (x[horizontal_axes[1]] - lower_bounds[horizontal_axes[1]]) / spans[
                horizontal_axes[1]
            ]
            for mode_x in range(num_modes):
                for mode_y in range(num_modes):
                    perturbation += (
                        coefficients[mode_x, mode_y]
                        * np.sin(2.0 * np.pi * (mode_x + 1) * xi)
                        * np.sin(2.0 * np.pi * (mode_y + 1) * zeta)
                    )

        return base_profile + amplitude * vertical_envelope * (
            perturbation / coefficient_scale
        )

    return _interpolator


class _ThermalConvectionProblem(TransientLinearProblem):
    def validate_boundary_conditions(self, domain_geom):
        velocity_boundary_field = self.config.boundary_field("velocity")
        temperature_boundary_field = self.config.boundary_field("temperature")

        validate_vector_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="velocity",
            boundary_field=velocity_boundary_field,
            domain_geom=domain_geom,
        )
        validate_scalar_standard_boundary_field(
            preset_name=self.spec.name,
            field_name="temperature",
            boundary_field=temperature_boundary_field,
            domain_geom=domain_geom,
        )

        expected_boundary_names = _expected_boundary_names(
            domain_geom.mesh.geometry.dim
        )
        actual_boundary_names = set(domain_geom.boundary_names)
        if actual_boundary_names != expected_boundary_names:
            raise ValueError(
                f"Preset '{self.spec.name}' requires standard boundary names "
                f"{sorted(expected_boundary_names)}. Got "
                f"{sorted(actual_boundary_names)}."
            )

        velocity_pairs = velocity_boundary_field.periodic_pair_keys()
        temperature_pairs = temperature_boundary_field.periodic_pair_keys()
        if velocity_pairs != temperature_pairs:
            raise ValueError(
                "Velocity and temperature boundary conditions must use identical "
                "periodic side pairs."
            )

    def setup(self) -> None:
        domain_geom = self.load_domain_geometry()
        self.domain_geom = domain_geom
        self.msh = domain_geom.mesh
        self.gdim = self.msh.geometry.dim
        self.vertical_axis = _vertical_axis(self.gdim)

        dt = self.config.time.dt
        k = int(self.config.parameters["k"])
        Ra = self.config.parameters["Ra"]
        Pr = self.config.parameters["Pr"]
        velocity_diffusivity = np.sqrt(Pr / Ra)
        temperature_diffusivity = 1.0 / np.sqrt(Ra * Pr)

        velocity_field = self.config.input("velocity")
        temperature_field = self.config.input("temperature")
        velocity_boundary_field = self.config.boundary_field("velocity")
        temperature_boundary_field = self.config.boundary_field("temperature")

        flow_system = create_taylor_hood_system(
            self.create_periodic_constraint,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
            pressure_degree=k,
            pressure_boundary_field=velocity_boundary_field,
        )

        S = fem.functionspace(self.msh, ("Lagrange", k + 1))
        temperature_bcs = apply_dirichlet_bcs(
            S,
            domain_geom,
            temperature_boundary_field,
            self.config.parameters,
        )
        temperature_mpc = self.create_periodic_constraint(
            S,
            domain_geom,
            temperature_boundary_field,
            temperature_bcs,
        )
        temperature_solution_space = (
            S if temperature_mpc is None else temperature_mpc.function_space
        )

        self._num_dofs = flow_system.num_dofs() + _space_num_dofs(
            temperature_solution_space
        )

        self.u_h = flow_system.create_velocity_function("velocity")
        self.p_h = flow_system.create_pressure_function("pressure")
        self.T_h = fem.Function(temperature_solution_space, name="temperature")
        self.u_n = flow_system.create_velocity_function("u_prev")
        self.T_n = fem.Function(temperature_solution_space, name="temperature_prev")

        assert velocity_field.initial_condition is not None
        apply_vector_ic(
            self.u_h,
            velocity_field.initial_condition,
            self.config.parameters,
            seed=self.config.seed,
        )
        self.u_h.x.scatter_forward()
        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()

        self.p_h.x.array[:] = 0.0
        self.p_h.x.scatter_forward()

        assert temperature_field.initial_condition is not None
        if temperature_field.initial_condition.type == "conductive_noise":
            amplitude = float(temperature_field.initial_condition.params["amplitude"])
            num_modes = int(temperature_field.initial_condition.params["num_modes"])
            self.T_h.interpolate(
                _conductive_noise_interpolator(
                    domain_geom=domain_geom,
                    temperature_boundary_field=temperature_boundary_field,
                    parameters=self.config.parameters,
                    amplitude=amplitude,
                    num_modes=num_modes,
                    seed=self.config.seed,
                )
            )
        else:
            apply_ic(
                self.T_h,
                temperature_field.initial_condition,
                self.config.parameters,
                seed=self.config.seed,
            )
        self.T_h.x.scatter_forward()
        self.T_n.x.array[:] = self.T_h.x.array
        self.T_n.x.scatter_forward()

        u = ufl.TrialFunction(flow_system.V)
        p = ufl.TrialFunction(flow_system.Q)
        T = ufl.TrialFunction(S)
        v = ufl.TestFunction(flow_system.V)
        q = ufl.TestFunction(flow_system.Q)
        s = ufl.TestFunction(S)

        delta_t = fem.Constant(self.msh, default_real_type(dt))
        zero_scalar = fem.Constant(self.msh, default_real_type(0.0))
        buoyancy_direction = np.zeros(self.gdim, dtype=default_real_type)
        buoyancy_direction[self.vertical_axis] = 1.0
        e_vertical = fem.Constant(self.msh, buoyancy_direction)

        a_blocks = [
            [
                ufl.inner(u / delta_t, v) * ufl.dx
                + ufl.inner(ufl.dot(ufl.grad(u), self.u_n), v) * ufl.dx
                + default_real_type(velocity_diffusivity)
                * ufl.inner(ufl.grad(u), ufl.grad(v))
                * ufl.dx,
                -ufl.inner(p, ufl.div(v)) * ufl.dx,
                -ufl.inner(T * e_vertical, v) * ufl.dx,
            ],
            [
                -ufl.inner(ufl.div(u), q) * ufl.dx,
                None,
                None,
            ],
            [
                None,
                None,
                ufl.inner(T / delta_t, s) * ufl.dx
                + ufl.inner(ufl.dot(ufl.grad(T), self.u_n), s) * ufl.dx
                + default_real_type(temperature_diffusivity)
                * ufl.inner(ufl.grad(T), ufl.grad(s))
                * ufl.dx,
            ],
        ]

        L_blocks = [
            ufl.inner(self.u_n / delta_t, v) * ufl.dx,
            ufl.inner(zero_scalar, q) * ufl.dx,
            ufl.inner(self.T_n / delta_t, s) * ufl.dx,
        ]

        assert velocity_field.source is not None
        velocity_source_form = build_vector_source_form(
            v,
            self.msh,
            velocity_field.source,
            self.config.parameters,
        )
        if velocity_source_form is not None:
            L_blocks[0] = L_blocks[0] + velocity_source_form

        traction_form = build_vector_natural_bc_forms(
            v,
            domain_geom,
            velocity_boundary_field,
            self.config.parameters,
        )
        if traction_form is not None:
            L_blocks[0] = L_blocks[0] + traction_form

        assert temperature_field.source is not None
        temperature_source_form = build_source_form(
            s,
            self.msh,
            temperature_field.source,
            self.config.parameters,
        )
        if temperature_source_form is not None:
            L_blocks[2] = L_blocks[2] + temperature_source_form

        a_temp_bc, L_temp_bc = build_natural_bc_forms(
            T,
            s,
            domain_geom,
            temperature_boundary_field,
            self.config.parameters,
        )
        if a_temp_bc is not None:
            a_blocks[2][2] = a_blocks[2][2] + a_temp_bc
        if L_temp_bc is not None:
            L_blocks[2] = L_blocks[2] + L_temp_bc

        bcs = flow_system.bcs + temperature_bcs
        if flow_system.mpc_u is None:
            mpc = None
        else:
            assert flow_system.mpc_p is not None
            assert temperature_mpc is not None
            mpc = [flow_system.mpc_u, flow_system.mpc_p, temperature_mpc]

        self._problem = self.create_linear_problem(
            a_blocks,
            L_blocks,
            u=[self.u_h, self.p_h, self.T_h],
            bcs=bcs,
            petsc_options_prefix="plm_thermal_convection_",
            kind="mpi",
            mpc=mpc,
        )

    def step(self, t: float, dt: float) -> bool:
        self._problem.solve()
        normalize_pressure(self.msh, self.p_h)

        self.u_n.x.array[:] = self.u_h.x.array
        self.u_n.x.scatter_forward()
        self.T_n.x.array[:] = self.T_h.x.array
        self.T_n.x.scatter_forward()

        return self._problem.solver.getConvergedReason() > 0

    def get_output_fields(self) -> dict[str, fem.Function]:
        return {
            "velocity": self.u_h,
            "pressure": self.p_h,
            "temperature": self.T_h,
        }

    def get_num_dofs(self) -> int:
        return self._num_dofs


@register_preset("thermal_convection")
class ThermalConvectionPreset(PDEPreset):
    @property
    def spec(self) -> PresetSpec:
        return _THERMAL_CONVECTION_SPEC

    def build_problem(self, config) -> ProblemInstance:
        return _ThermalConvectionProblem(self.spec, config)
