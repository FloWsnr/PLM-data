import textwrap
import time

import numpy as np

from plm_data.core.config import (
    BoundaryFieldConfig,
    FieldExpressionConfig,
    SimulationConfig,
)
from plm_data.core.initial_conditions import (
    build_scalar_ic_interpolator,
    build_vector_ic_interpolator,
)
from plm_data.core.spatial_fields import (
    component_labels_for_dim,
    is_exact_zero_field_expression,
)
from plm_data.presets.base import RunResult

from .base import _OpenFOAMProblemBase
from .core import (
    _OPENFOAM_UNIVERSAL_GAS_CONSTANT,
    _ensure_supported_expression,
    _expected_boundary_names,
    _foam_dict,
    _format_scalar,
    _is_uniform_scalar_expression,
    _normalise_patch_name,
    _pad_to_foam_vector,
    _periodic_patch_name,
    _resolve_periodic_vectors,
    _scalar_field_cpp_expr,
    _single_boundary_entry,
    _validate_side_names,
    _write_text,
)
from .fields import (
    _scalar_fixed_value_block,
    _vector_fixed_value_block,
    _write_scalar_field_file,
    _write_vector_field_file,
)


def _temperature_from_density_pressure_block(
    *,
    density_expr: FieldExpressionConfig,
    pressure_expr: FieldExpressionConfig,
    gas_constant: float,
    parameters: dict[str, float],
    gdim: int,
    field_name: str,
) -> str:
    density_uniform = _is_uniform_scalar_expression(
        density_expr,
        parameters=parameters,
    )
    pressure_uniform = _is_uniform_scalar_expression(
        pressure_expr,
        parameters=parameters,
    )
    if density_uniform is not None and pressure_uniform is not None:
        return textwrap.dedent(
            f"""\
            type            fixedValue;
            value           uniform {_format_scalar(pressure_uniform / (gas_constant * density_uniform))};
            """
        )

    density_cpp = _scalar_field_cpp_expr(
        density_expr,
        parameters=parameters,
        gdim=gdim,
    )
    pressure_cpp = _scalar_field_cpp_expr(
        pressure_expr,
        parameters=parameters,
        gdim=gdim,
    )
    temperature_cpp = (
        f"(({pressure_cpp})/({_format_scalar(gas_constant)}*({density_cpp})))"
    )
    return textwrap.dedent(
        f"""\
        type            codedFixedValue;
        name            {field_name}_coded;
        code
        #{{
            const auto& faceCentres = this->patch().Cf();
            scalarField values(faceCentres.size(), 0.0);
            forAll(faceCentres, facei)
            {{
                const scalar x = faceCentres[facei].x();
                const scalar y = faceCentres[facei].y();
                const scalar z = faceCentres[facei].z();
                values[facei] = {temperature_cpp};
            }}
            operator==(values);
        #}};
        value           uniform 0;
        """
    )


class OpenFOAMEulerProblem(_OpenFOAMProblemBase):
    solver_name = "shockFluid"

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._assert_supported_mesh_domain()
        if self._gdim not in {2, 3}:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D/3D, got {self._gdim}D."
            )

        self._positive_parameter("gas_constant")
        self._positive_parameter("c_v")
        self._scalar_degree()

        density_field = self.config.input("density")
        velocity_field = self.config.input("velocity")
        pressure_field = self.config.input("pressure")

        for context, expr in (
            ("inputs.density.initial_condition", density_field.initial_condition),
            ("inputs.velocity.initial_condition", velocity_field.initial_condition),
            ("inputs.pressure.initial_condition", pressure_field.initial_condition),
        ):
            _ensure_supported_expression(
                expr,
                allow_quadrants=True,
                allow_initial_only_types=True,
                context=context,
            )

        for field_name, expr in (
            ("density", density_field.initial_condition),
            ("velocity", velocity_field.initial_condition),
            ("pressure", pressure_field.initial_condition),
        ):
            if expr.type == "custom":
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} does not support "
                    f"{field_name}.initial_condition.type='custom'."
                )

    def _positive_parameter(self, name: str) -> float:
        value = float(self.config.parameters[name])
        if value <= 0.0:
            raise ValueError(
                f"Preset '{self.spec.name}' requires parameter '{name}' > 0. "
                f"Got {value}."
            )
        return value

    def _scalar_degree(self) -> int:
        raw_degree = self.config.parameters["k"]
        degree = int(raw_degree)
        if float(degree) != float(raw_degree) or degree < 1:
            raise ValueError(
                f"Preset '{self.spec.name}' requires integer parameter 'k' >= 1. "
                f"Got {raw_degree}."
            )
        return degree

    def _gas_constant(self) -> float:
        return self._positive_parameter("gas_constant")

    def _heat_capacity_cv(self) -> float:
        return self._positive_parameter("c_v")

    def _molecular_weight(self) -> float:
        return _OPENFOAM_UNIVERSAL_GAS_CONSTANT / self._gas_constant()

    def _density_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("density")

    def _velocity_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("velocity")

    def _pressure_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("pressure")

    def _periodic_vectors(self) -> dict[frozenset[str], np.ndarray]:
        density_pairs = self._density_boundary().periodic_pair_keys()
        velocity_pairs = self._velocity_boundary().periodic_pair_keys()
        pressure_pairs = self._pressure_boundary().periodic_pair_keys()
        if density_pairs != velocity_pairs or density_pairs != pressure_pairs:
            raise ValueError(
                "Density, velocity, and pressure boundary conditions must use "
                "identical periodic side pairs."
            )
        return _resolve_periodic_vectors(self.config, density_pairs)

    def _validate_boundary_conditions(self) -> None:
        expected = _expected_boundary_names(self.config)
        for field_name, boundary_field in (
            ("density", self._density_boundary()),
            ("velocity", self._velocity_boundary()),
            ("pressure", self._pressure_boundary()),
        ):
            _validate_side_names(
                boundary_field=boundary_field,
                expected_sides=expected,
                preset_name=self.spec.name,
                field_name=field_name,
            )

        for side in expected:
            density_entry = _single_boundary_entry(
                boundary_field=self._density_boundary(),
                side=side,
            )
            velocity_entry = _single_boundary_entry(
                boundary_field=self._velocity_boundary(),
                side=side,
            )
            pressure_entry = _single_boundary_entry(
                boundary_field=self._pressure_boundary(),
                side=side,
            )

            if density_entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} only supports density "
                    f"dirichlet/neumann/periodic boundaries. Side '{side}' uses "
                    f"'{density_entry.type}'."
                )
            if pressure_entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} only supports pressure "
                    f"dirichlet/neumann/periodic boundaries. Side '{side}' uses "
                    f"'{pressure_entry.type}'."
                )
            if velocity_entry.type not in {"dirichlet", "neumann", "periodic", "slip"}:
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} only supports velocity "
                    "dirichlet, zero-style neumann, periodic, or slip "
                    f"boundaries. Side '{side}' uses '{velocity_entry.type}'."
                )

            if density_entry.type != pressure_entry.type:
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} requires density and "
                    f"pressure to use the same operator on side '{side}'."
                )

            if density_entry.value is not None:
                _ensure_supported_expression(
                    density_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.density.{side}",
                )
            if pressure_entry.value is not None:
                _ensure_supported_expression(
                    pressure_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.pressure.{side}",
                )
            if velocity_entry.value is not None:
                _ensure_supported_expression(
                    velocity_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.velocity.{side}",
                )

            if density_entry.type == "neumann" and density_entry.value is not None:
                if not is_exact_zero_field_expression(
                    density_entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        f"OpenFOAM-backed {self.spec.name} currently only supports "
                        "zero-gradient style density neumann conditions. Side "
                        f"'{side}' is non-zero."
                    )
            if pressure_entry.type == "neumann" and pressure_entry.value is not None:
                if not is_exact_zero_field_expression(
                    pressure_entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        f"OpenFOAM-backed {self.spec.name} currently only supports "
                        "zero-gradient style pressure neumann conditions. Side "
                        f"'{side}' is non-zero."
                    )
            if velocity_entry.type == "neumann" and velocity_entry.value is not None:
                if not is_exact_zero_field_expression(
                    velocity_entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        f"OpenFOAM-backed {self.spec.name} currently only supports "
                        "zero-gradient style velocity neumann conditions. Side "
                        f"'{side}' is non-zero."
                    )

    def _write_control_dict(self) -> None:
        if self.config.time is None:
            raise ValueError(f"Preset '{self.spec.name}' requires a time section.")
        interval = (
            self.config.time.t_end
            if self.config.output.num_frames <= 1
            else self.config.time.t_end / float(self.config.output.num_frames - 1)
        )
        content = _foam_dict(
            object_name="controlDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                solver          {self.solver_name};

                startFrom       startTime;
                startTime       0;
                stopAt          endTime;
                endTime         {_format_scalar(self.config.time.t_end)};
                deltaT          {_format_scalar(self.config.time.dt)};

                writeControl    runTime;
                writeInterval   {_format_scalar(interval)};
                purgeWrite      0;

                writeFormat     ascii;
                writePrecision  12;
                writeCompression off;
                timeFormat      general;
                timePrecision   12;
                runTimeModifiable false;

                adjustTimeStep  no;
                maxCo           0.1;
                maxDeltaT       {_format_scalar(self.config.time.dt)};
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "controlDict", content)

    def _write_physical_properties(self) -> None:
        content = _foam_dict(
            object_name="physicalProperties",
            location="constant",
            body=textwrap.dedent(
                f"""\
                thermoType
                {{
                    type            hePsiThermo;
                    mixture         pureMixture;
                    transport       const;
                    thermo          eConst;
                    equationOfState perfectGas;
                    specie          specie;
                    energy          sensibleInternalEnergy;
                }}

                mixture
                {{
                    specie
                    {{
                        molWeight       {_format_scalar(self._molecular_weight())};
                    }}
                    thermodynamics
                    {{
                        Cv              {_format_scalar(self._heat_capacity_cv())};
                        hf              0;
                    }}
                    transport
                    {{
                        mu              0;
                        kappa           0;
                    }}
                }}
                """
            ),
        )
        _write_text(
            self._current_case_dir() / "constant" / "physicalProperties",
            content,
        )

    def _write_fv_schemes(self) -> None:
        content = _foam_dict(
            object_name="fvSchemes",
            location="system",
            body=textwrap.dedent(
                """\
                fluxScheme      Kurganov;

                ddtSchemes
                {
                    default         Euler;
                }

                gradSchemes
                {
                    default         Gauss linear;
                }

                divSchemes
                {
                    default         none;
                }

                laplacianSchemes
                {
                    default         Gauss linear orthogonal;
                }

                interpolationSchemes
                {
                    default         none;

                    reconstruct(rho) vanAlbada;
                    reconstruct(U)   vanAlbadaV;
                    reconstruct(T)   vanAlbada;

                    flux(rhoU)       linear;
                }

                snGradSchemes
                {
                    default         orthogonal;
                }
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "fvSchemes", content)

    def _write_fv_solution(self) -> None:
        content = _foam_dict(
            object_name="fvSolution",
            location="system",
            body=textwrap.dedent(
                """\
                solvers
                {
                    "(rho|rhoU|rhoE).*"
                    {
                        solver          diagonal;
                    }

                    "U.*"
                    {
                        solver          smoothSolver;
                        smoother        GaussSeidel;
                        nSweeps         2;
                        tolerance       1e-09;
                        relTol          0.01;
                    }

                    "e.*"
                    {
                        $U;
                        tolerance       1e-10;
                        relTol          0;
                    }
                }

                PIMPLE
                {
                    nOuterCorrectors 2;
                }
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "fvSolution", content)

    def _field_boundary_entries(
        self,
        *,
        periodic_pairs: dict[frozenset[str], np.ndarray],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        density_boundary = self._density_boundary()
        velocity_boundary = self._velocity_boundary()
        pressure_boundary = self._pressure_boundary()

        velocity_entries: dict[str, str] = {}
        pressure_entries: dict[str, str] = {}
        temperature_entries: dict[str, str] = {}

        for side in density_boundary.sides:
            density_entry = _single_boundary_entry(
                boundary_field=density_boundary,
                side=side,
            )
            velocity_entry = _single_boundary_entry(
                boundary_field=velocity_boundary,
                side=side,
            )
            pressure_entry = _single_boundary_entry(
                boundary_field=pressure_boundary,
                side=side,
            )
            patch_name = _periodic_patch_name(side, periodic_pairs)

            if density_entry.type == "periodic":
                pair_key = frozenset({side, density_entry.pair_with})
                if pair_key not in periodic_pairs:
                    raise AssertionError(
                        f"Missing periodic pair metadata for side '{side}'."
                    )
                cyclic_entry = "type            cyclic;"
                velocity_entries[_normalise_patch_name(side)] = cyclic_entry
                pressure_entries[_normalise_patch_name(side)] = cyclic_entry
                temperature_entries[_normalise_patch_name(side)] = cyclic_entry
                continue

            if density_entry.type == "dirichlet":
                assert density_entry.value is not None
                assert pressure_entry.value is not None
                pressure_entries[patch_name] = _scalar_fixed_value_block(
                    expr=pressure_entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"p_{patch_name}",
                )
                temperature_entries[patch_name] = (
                    _temperature_from_density_pressure_block(
                        density_expr=density_entry.value,
                        pressure_expr=pressure_entry.value,
                        gas_constant=self._gas_constant(),
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"T_{patch_name}",
                    )
                )
            elif density_entry.type == "neumann":
                pressure_entries[patch_name] = "type            zeroGradient;"
                temperature_entries[patch_name] = "type            zeroGradient;"
            else:
                raise AssertionError(
                    f"Unhandled density boundary type '{density_entry.type}'."
                )

            if velocity_entry.type == "dirichlet":
                assert velocity_entry.value is not None
                velocity_entries[patch_name] = _vector_fixed_value_block(
                    expr=velocity_entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"U_{patch_name}",
                )
            elif velocity_entry.type == "neumann":
                velocity_entries[patch_name] = "type            zeroGradient;"
            elif velocity_entry.type == "slip":
                velocity_entries[patch_name] = "type            slip;"
            else:
                raise AssertionError(
                    f"Unhandled velocity boundary type '{velocity_entry.type}'."
                )

        if self._gdim == 2:
            velocity_entries["frontAndBack"] = "type            empty;"
            pressure_entries["frontAndBack"] = "type            empty;"
            temperature_entries["frontAndBack"] = "type            empty;"

        return velocity_entries, pressure_entries, temperature_entries

    def _write_initial_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        temperature_entries: dict[str, str],
        cell_centres: np.ndarray,
    ) -> None:
        density_ic = build_scalar_ic_interpolator(
            self.config.input("density").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="density",
        )
        velocity_ic = build_vector_ic_interpolator(
            self.config.input("velocity").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="velocity",
        )
        pressure_ic = build_scalar_ic_interpolator(
            self.config.input("pressure").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="pressure",
        )
        if density_ic is None or velocity_ic is None or pressure_ic is None:
            raise ValueError(
                f"OpenFOAM-backed {self.spec.name} requires explicit initial "
                "condition interpolators for density, velocity, and pressure."
            )

        density_values = np.asarray(density_ic(cell_centres), dtype=float)
        velocity_values = _pad_to_foam_vector(
            np.asarray(velocity_ic(cell_centres), dtype=float).T
        )
        pressure_values = np.asarray(pressure_ic(cell_centres), dtype=float)
        temperature_values = pressure_values / (self._gas_constant() * density_values)

        if np.min(density_values) <= 0.0:
            raise ValueError(
                f"OpenFOAM-backed {self.spec.name} requires strictly positive "
                "initial density."
            )
        if np.min(pressure_values) <= 0.0:
            raise ValueError(
                f"OpenFOAM-backed {self.spec.name} requires strictly positive "
                "initial pressure."
            )
        if np.min(temperature_values) <= 0.0:
            raise ValueError(
                f"OpenFOAM-backed {self.spec.name} requires strictly positive "
                "initial temperature."
            )

        _write_vector_field_file(
            self._current_case_dir() / "0" / "U",
            object_name="U",
            dimensions="[0 1 -1 0 0 0 0]",
            internal_values=velocity_values,
            internal_uniform=None,
            boundary_entries=velocity_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "p",
            object_name="p",
            dimensions="[1 -1 -2 0 0 0 0]",
            internal_values=pressure_values,
            internal_uniform=None,
            boundary_entries=pressure_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "T",
            object_name="T",
            dimensions="[0 0 0 1 0 0 0]",
            internal_values=temperature_values,
            internal_uniform=None,
            boundary_entries=temperature_entries,
        )

    def run(self, output) -> RunResult:
        self._validate_boundary_conditions()
        case_dir = output.output_dir / "openfoam_case"
        self._prepare_case_dir(output.output_dir)
        if "vtk" in self.config.output.formats:
            output.use_external_paraview_case(case_dir)

        periodic_pairs = self._periodic_vectors()
        self._write_control_dict()
        self._write_physical_properties()
        self._write_momentum_transport()
        self._write_fv_schemes()
        self._write_fv_solution()
        self._create_decompose_par_dict()
        self._build_mesh(periodic_pairs=periodic_pairs)
        (
            velocity_entries,
            pressure_entries,
            temperature_entries,
        ) = self._field_boundary_entries(periodic_pairs=periodic_pairs)
        cell_centres = self._read_internal_cell_centres()
        self._write_initial_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            temperature_entries=temperature_entries,
            cell_centres=cell_centres,
        )

        run_start = time.perf_counter()
        self._solve_openfoam_case()
        solve_seconds = time.perf_counter() - run_start

        num_cells, num_steps = self._sample_case_to_output(
            output=output,
            field_map={
                "density": "rho",
                "pressure": "p",
                "temperature": "T",
                **{
                    f"velocity_{label}": "U"
                    for label in component_labels_for_dim(self._gdim)
                },
            },
            normalize_pressure_field=None,
            density_from_pressure_temperature_gas_constant=self._gas_constant(),
        )
        num_dofs = num_cells * (self._gdim + 2)
        diagnostics = {
            "solver_health": {"status": "pass", "applied": False, "checks": {}},
            "runtime_health": {"status": "pass", "applied": False, "checks": {}},
            "openfoam": {
                "case_dir": str(case_dir),
                "solver": self.solver_name,
                "num_subdomains": self._n_subdomains,
                "solve_seconds": solve_seconds,
            },
        }
        return RunResult(
            num_dofs=num_dofs,
            solver_converged=True,
            wall_time=solve_seconds,
            num_timesteps=num_steps,
            diagnostics=diagnostics,
        )
