import math
import textwrap
import time

import numpy as np

from plm_data.core.config import BoundaryFieldConfig, SimulationConfig
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
    _ensure_supported_expression,
    _expected_boundary_names,
    _foam_dict,
    _format_scalar,
    _normalise_patch_name,
    _pad_to_foam_vector,
    _periodic_patch_name,
    _resolve_periodic_vectors,
    _single_boundary_entry,
    _validate_side_names,
    _vector_field_cpp_expr,
    _write_text,
)
from .fields import (
    _scalar_field_cpp_expr,
    _scalar_fixed_gradient_block,
    _scalar_fixed_value_block,
    _vector_fixed_value_block,
    _write_scalar_field_file,
    _write_vector_field_file,
)


class OpenFOAMNavierStokesProblem(_OpenFOAMProblemBase):
    solver_name = "incompressibleFluid"

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._assert_supported_mesh_domain()
        if self._gdim not in {2, 3}:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D/3D, got {self._gdim}D."
            )
        velocity_field = self.config.input("velocity")
        _ensure_supported_expression(
            velocity_field.initial_condition,
            allow_quadrants=True,
            allow_initial_only_types=True,
            context="inputs.velocity.initial_condition",
        )
        if velocity_field.source is not None:
            _ensure_supported_expression(
                velocity_field.source,
                allow_quadrants=False,
                context="inputs.velocity.source",
            )
        if velocity_field.initial_condition.type == "custom":
            raise ValueError(
                "OpenFOAM-backed Navier-Stokes does not support "
                "velocity.initial_condition.type='custom'."
            )

    def _velocity_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("velocity")

    def _periodic_vectors(self) -> dict[frozenset[str], np.ndarray]:
        return _resolve_periodic_vectors(
            self.config,
            self._velocity_boundary().periodic_pair_keys(),
        )

    def _validate_boundary_conditions(self) -> None:
        velocity_boundary = self._velocity_boundary()
        _validate_side_names(
            boundary_field=velocity_boundary,
            expected_sides=_expected_boundary_names(self.config),
            preset_name=self.spec.name,
            field_name="velocity",
        )

        for side in velocity_boundary.sides:
            entry = _single_boundary_entry(boundary_field=velocity_boundary, side=side)
            if entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    f"OpenFOAM-backed Navier-Stokes does not support velocity "
                    f"boundary operator '{entry.type}' on side '{side}'."
                )
            if entry.value is not None:
                _ensure_supported_expression(
                    entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.velocity.{side}",
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
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "controlDict", content)

    def _write_physical_properties(self) -> None:
        reynolds = float(self.config.parameters["Re"])
        content = _foam_dict(
            object_name="physicalProperties",
            location="constant",
            body=textwrap.dedent(
                f"""\
                viscosityModel  constant;

                nu              {_format_scalar(1.0 / reynolds)} [m^2/s];
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
                    div(phi,U)      Gauss linearUpwind grad(U);
                    div((nuEff*dev2(T(grad(U))))) Gauss linear;
                }

                laplacianSchemes
                {
                    default         Gauss linear corrected;
                }

                interpolationSchemes
                {
                    default         linear;
                }

                snGradSchemes
                {
                    default         corrected;
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
                    p
                    {
                        solver          GAMG;
                        tolerance       1e-8;
                        relTol          0.05;
                        smoother        GaussSeidel;
                        nCellsInCoarsestLevel 1;
                    }

                    pFinal
                    {
                        $p;
                        relTol          0;
                    }

                    U
                    {
                        solver          PBiCGStab;
                        preconditioner  DILU;
                        tolerance       1e-9;
                        relTol          0;
                    }

                    UFinal
                    {
                        $U;
                        relTol          0;
                    }
                }

                PIMPLE
                {
                    nCorrectors                 2;
                    nNonOrthogonalCorrectors    0;
                    pRefCell                    0;
                    pRefValue                   0;
                }
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "fvSolution", content)

    def _velocity_patch_entries(
        self,
        *,
        periodic_pairs: dict[frozenset[str], np.ndarray],
    ) -> tuple[dict[str, str], dict[str, str]]:
        boundary_field = self._velocity_boundary()
        velocity_entries: dict[str, str] = {}
        pressure_entries: dict[str, str] = {}

        for side in boundary_field.sides:
            entry = _single_boundary_entry(boundary_field=boundary_field, side=side)
            patch_name = _periodic_patch_name(side, periodic_pairs)
            if entry.type == "periodic":
                pair_key = frozenset({side, entry.pair_with})
                if pair_key not in periodic_pairs:
                    raise AssertionError(
                        f"Missing OpenFOAM periodic pair metadata for side '{side}'."
                    )
                cyclic_entry = "type            cyclic;"
                velocity_entries[_normalise_patch_name(side)] = cyclic_entry
                pressure_entries[_normalise_patch_name(side)] = cyclic_entry
                continue

            assert entry.value is not None
            if entry.type == "dirichlet":
                velocity_entries[patch_name] = _vector_fixed_value_block(
                    expr=entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"U_{patch_name}",
                    zero_patch_type="noSlip",
                )
                pressure_entries[patch_name] = "type            zeroGradient;"
                continue

            if entry.type == "neumann":
                if not is_exact_zero_field_expression(
                    entry.value, self.config.parameters
                ):
                    raise ValueError(
                        "OpenFOAM-backed Navier-Stokes currently only supports "
                        "zero traction / zero-gradient style velocity neumann "
                        f"conditions. Side '{side}' is non-zero."
                    )
                velocity_entries[patch_name] = textwrap.dedent(
                    """\
                    type            pressureInletOutletVelocity;
                    value           uniform (0 0 0);
                    """
                )
                pressure_entries[patch_name] = textwrap.dedent(
                    """\
                    type            fixedValue;
                    value           uniform 0;
                    """
                )
                continue

            raise AssertionError(f"Unhandled velocity boundary type '{entry.type}'.")

        if self._gdim == 2:
            velocity_entries["frontAndBack"] = "type            empty;"
            pressure_entries["frontAndBack"] = "type            empty;"
        return velocity_entries, pressure_entries

    def _write_sources(self) -> None:
        velocity_source = self.config.input("velocity").source
        body = ""
        if velocity_source is not None and not is_exact_zero_field_expression(
            velocity_source,
            self.config.parameters,
        ):
            _ensure_supported_expression(
                velocity_source,
                allow_quadrants=False,
                context="inputs.velocity.source",
            )
            vector_expr = _vector_field_cpp_expr(
                velocity_source,
                parameters=self.config.parameters,
                gdim=self._gdim,
            )
            body = textwrap.dedent(
                f"""\
                velocitySource
                {{
                    type            coded;
                    cellZone        all;
                    field           U;
                    codeAddSup
                    #{{
                        const auto& C = mesh().C();
                        const auto& V = mesh().V();
                        auto& source = eqn.source();
                        forAll(source, celli)
                        {{
                            const scalar x = C[celli].x();
                            const scalar y = C[celli].y();
                            const scalar z = C[celli].z();
                            source[celli] -= {vector_expr}*V[celli];
                        }}
                    #}};
                }}
                """
            )
        content = _foam_dict(
            object_name="fvModels",
            location="constant",
            body=body or "",
        )
        _write_text(self._current_case_dir() / "constant" / "fvModels", content)

    def _write_initial_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        cell_centres: np.ndarray,
    ) -> None:
        velocity_ic = build_vector_ic_interpolator(
            self.config.input("velocity").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="velocity",
        )
        if velocity_ic is None:
            raise ValueError(
                "OpenFOAM-backed Navier-Stokes requires an explicit velocity "
                "initial-condition interpolator."
            )
        velocity_values = _pad_to_foam_vector(
            np.asarray(velocity_ic(cell_centres), dtype=float).T
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
            dimensions="[0 2 -2 0 0 0 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=pressure_entries,
        )

    def _write_placeholder_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
    ) -> None:
        _write_vector_field_file(
            self._current_case_dir() / "0" / "U",
            object_name="U",
            dimensions="[0 1 -1 0 0 0 0]",
            internal_values=None,
            internal_uniform=np.zeros(3, dtype=float),
            boundary_entries=velocity_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "p",
            object_name="p",
            dimensions="[0 2 -2 0 0 0 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=pressure_entries,
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
        velocity_entries, pressure_entries = self._velocity_patch_entries(
            periodic_pairs=periodic_pairs,
        )
        self._write_placeholder_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
        )
        cell_centres = self._read_internal_cell_centres()
        self._write_initial_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            cell_centres=cell_centres,
        )
        self._write_sources()

        run_start = time.perf_counter()
        self._solve_openfoam_case()
        solve_seconds = time.perf_counter() - run_start

        num_cells, num_steps = self._sample_case_to_output(
            output=output,
            field_map={
                "pressure": "p",
                **{
                    f"velocity_{label}": "U"
                    for label in component_labels_for_dim(self._gdim)
                },
            },
            normalize_pressure_field="pressure",
        )
        num_dofs = num_cells * (self._gdim + 1)
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


class OpenFOAMThermalConvectionProblem(_OpenFOAMProblemBase):
    solver_name = "incompressibleFluid"

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._assert_supported_mesh_domain()
        if self._gdim not in {2, 3}:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D/3D, got {self._gdim}D."
            )
        velocity_field = self.config.input("velocity")
        temperature_field = self.config.input("temperature")
        _ensure_supported_expression(
            velocity_field.initial_condition,
            allow_quadrants=True,
            allow_initial_only_types=True,
            context="inputs.velocity.initial_condition",
        )
        _ensure_supported_expression(
            temperature_field.initial_condition,
            allow_quadrants=True,
            allow_initial_only_types=True,
            context="inputs.temperature.initial_condition",
        )
        if velocity_field.source is not None:
            _ensure_supported_expression(
                velocity_field.source,
                allow_quadrants=False,
                context="inputs.velocity.source",
            )
        if temperature_field.source is not None:
            _ensure_supported_expression(
                temperature_field.source,
                allow_quadrants=False,
                context="inputs.temperature.source",
            )
        if velocity_field.initial_condition.type == "custom":
            raise ValueError(
                "OpenFOAM-backed thermal_convection does not support "
                "velocity.initial_condition.type='custom'."
            )
        if temperature_field.initial_condition.type == "custom":
            raise ValueError(
                "OpenFOAM-backed thermal_convection does not support "
                "temperature.initial_condition.type='custom'."
            )

    def _velocity_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("velocity")

    def _temperature_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("temperature")

    def _temperature_gradient_scale(self) -> float:
        return math.sqrt(
            float(self.config.parameters["Ra"]) * float(self.config.parameters["Pr"])
        )

    def _periodic_vectors(self) -> dict[frozenset[str], np.ndarray]:
        velocity_pairs = self._velocity_boundary().periodic_pair_keys()
        temperature_pairs = self._temperature_boundary().periodic_pair_keys()
        if velocity_pairs != temperature_pairs:
            raise ValueError(
                "Velocity and temperature boundary conditions must use identical "
                "periodic side pairs."
            )
        return _resolve_periodic_vectors(self.config, velocity_pairs)

    def _validate_boundary_conditions(self) -> None:
        expected = _expected_boundary_names(self.config)
        _validate_side_names(
            boundary_field=self._velocity_boundary(),
            expected_sides=expected,
            preset_name=self.spec.name,
            field_name="velocity",
        )
        _validate_side_names(
            boundary_field=self._temperature_boundary(),
            expected_sides=expected,
            preset_name=self.spec.name,
            field_name="temperature",
        )
        for side in expected:
            velocity_entry = _single_boundary_entry(
                boundary_field=self._velocity_boundary(),
                side=side,
            )
            temperature_entry = _single_boundary_entry(
                boundary_field=self._temperature_boundary(),
                side=side,
            )
            if velocity_entry.type not in {"dirichlet", "periodic"}:
                raise ValueError(
                    f"OpenFOAM-backed thermal_convection does not support velocity "
                    f"boundary operator '{velocity_entry.type}' on side '{side}'."
                )
            if temperature_entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    f"OpenFOAM-backed thermal_convection does not support "
                    f"temperature boundary operator '{temperature_entry.type}' on "
                    f"side '{side}'."
                )
            if velocity_entry.value is not None:
                _ensure_supported_expression(
                    velocity_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.velocity.{side}",
                )
            if temperature_entry.value is not None:
                _ensure_supported_expression(
                    temperature_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.temperature.{side}",
                )

    def _write_control_dict(self) -> None:
        if self.config.time is None:
            raise ValueError(f"Preset '{self.spec.name}' requires a time section.")
        temperature_diffusivity = 1.0 / math.sqrt(
            float(self.config.parameters["Ra"]) * float(self.config.parameters["Pr"])
        )
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

                functions
                {{
                    temperatureTransport
                    {{
                        type            scalarTransport;
                        libs            ("libsolverFunctionObjects.so");
                        field           T;
                        phi             phi;
                        schemesField    T;
                        diffusivity     constant;
                        D               {_format_scalar(temperature_diffusivity)};
                        nCorr           1;
                        executeControl  timeStep;
                        executeInterval 1;
                        writeControl    writeTime;
                        writeInterval   1;
                    }}
                }}
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "controlDict", content)

    def _write_physical_properties(self) -> None:
        rayleigh = float(self.config.parameters["Ra"])
        prandtl = float(self.config.parameters["Pr"])
        kinematic_viscosity = math.sqrt(prandtl / rayleigh)
        content = _foam_dict(
            object_name="physicalProperties",
            location="constant",
            body=textwrap.dedent(
                f"""\
                viscosityModel  constant;

                nu              {_format_scalar(kinematic_viscosity)} [m^2/s];
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
                    default                 none;
                    div(phi,U)              Gauss linearUpwind grad(U);
                    div(phi,T)              Gauss upwind;
                    div((nuEff*dev2(T(grad(U))))) Gauss linear;
                }

                laplacianSchemes
                {
                    default         Gauss linear corrected;
                }

                interpolationSchemes
                {
                    default         linear;
                }

                snGradSchemes
                {
                    default         corrected;
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
                    p
                    {
                        solver          GAMG;
                        smoother        GaussSeidel;
                        tolerance       1e-8;
                        relTol          0.05;
                        nCellsInCoarsestLevel 1;
                    }

                    pFinal
                    {
                        $p;
                        relTol          0;
                    }

                    U
                    {
                        solver          PBiCGStab;
                        preconditioner  DILU;
                        tolerance       1e-9;
                        relTol          0;
                    }

                    UFinal
                    {
                        $U;
                        relTol          0;
                    }

                    T
                    {
                        solver          PBiCGStab;
                        preconditioner  DILU;
                        tolerance       1e-9;
                        relTol          0;
                    }
                }

                PIMPLE
                {
                    momentumPredictor          yes;
                    nCorrectors                2;
                    nNonOrthogonalCorrectors   0;
                    pRefCell                   0;
                    pRefValue                  0;
                }

                relaxationFactors
                {
                    equations
                    {
                        ".*"   1.0;
                    }
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
        velocity_entries: dict[str, str] = {}
        pressure_entries: dict[str, str] = {}
        temperature_entries: dict[str, str] = {}

        velocity_boundary = self._velocity_boundary()
        temperature_boundary = self._temperature_boundary()
        temperature_gradient_scale = self._temperature_gradient_scale()
        for side in velocity_boundary.sides:
            velocity_entry = _single_boundary_entry(
                boundary_field=velocity_boundary,
                side=side,
            )
            temperature_entry = _single_boundary_entry(
                boundary_field=temperature_boundary,
                side=side,
            )
            patch_name = _periodic_patch_name(side, periodic_pairs)
            if velocity_entry.type == "periodic":
                pair_key = frozenset({side, velocity_entry.pair_with})
                if pair_key not in periodic_pairs:
                    raise AssertionError(
                        f"Missing periodic pair metadata for side '{side}'."
                    )
                cyclic_entry = "type            cyclic;"
                velocity_entries[_normalise_patch_name(side)] = cyclic_entry
                pressure_entries[_normalise_patch_name(side)] = cyclic_entry
                temperature_entries[_normalise_patch_name(side)] = cyclic_entry
                continue

            assert velocity_entry.value is not None
            velocity_entries[patch_name] = _vector_fixed_value_block(
                expr=velocity_entry.value,
                parameters=self.config.parameters,
                gdim=self._gdim,
                field_name=f"U_{patch_name}",
                zero_patch_type="noSlip",
            )
            pressure_entries[patch_name] = "type            zeroGradient;"

            if temperature_entry.type == "dirichlet":
                assert temperature_entry.value is not None
                temperature_entries[patch_name] = _scalar_fixed_value_block(
                    expr=temperature_entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"T_{patch_name}",
                )
                continue

            if temperature_entry.type == "neumann":
                assert temperature_entry.value is not None
                temperature_entries[patch_name] = _scalar_fixed_gradient_block(
                    expr=temperature_entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"T_{patch_name}",
                    gradient_scale=temperature_gradient_scale,
                )
                continue

            raise AssertionError(
                f"Unhandled temperature boundary type '{temperature_entry.type}'."
            )

        if self._gdim == 2:
            velocity_entries["frontAndBack"] = "type            empty;"
            pressure_entries["frontAndBack"] = "type            empty;"
            temperature_entries["frontAndBack"] = "type            empty;"
        return velocity_entries, pressure_entries, temperature_entries

    def _write_sources(self) -> None:
        blocks: list[str] = []
        vertical_axis = 1 if self._gdim == 2 else 2
        buoyancy_components = ["0.0", "0.0", "0.0"]
        buoyancy_components[vertical_axis] = "temperature[celli]"
        buoyancy_vector = "Foam::vector(" + ", ".join(buoyancy_components) + ")"
        blocks.append(
            textwrap.dedent(
                f"""\
                buoyancySource
                {{
                    type            coded;
                    cellZone        all;
                    field           U;
                    codeAddSup
                    #{{
                        const auto& temperature =
                            mesh().lookupObject<volScalarField>("T");
                        const auto& V = mesh().V();
                        auto& source = eqn.source();
                        forAll(source, celli)
                        {{
                            source[celli] -= {buoyancy_vector}*V[celli];
                        }}
                    #}};
                }}
                """
            ).rstrip()
        )
        velocity_source = self.config.input("velocity").source
        if velocity_source is not None and not is_exact_zero_field_expression(
            velocity_source,
            self.config.parameters,
        ):
            vector_expr = _vector_field_cpp_expr(
                velocity_source,
                parameters=self.config.parameters,
                gdim=self._gdim,
            )
            blocks.append(
                textwrap.dedent(
                    f"""\
                    velocitySource
                    {{
                        type            coded;
                        cellZone        all;
                        field           U;
                        codeAddSup
                        #{{
                            const auto& C = mesh().C();
                            const auto& V = mesh().V();
                            auto& source = eqn.source();
                            forAll(source, celli)
                            {{
                                const scalar x = C[celli].x();
                                const scalar y = C[celli].y();
                                const scalar z = C[celli].z();
                                source[celli] -= {vector_expr}*V[celli];
                            }}
                        #}};
                    }}
                    """
                ).rstrip()
            )

        temperature_source = self.config.input("temperature").source
        if temperature_source is not None and not is_exact_zero_field_expression(
            temperature_source,
            self.config.parameters,
        ):
            scalar_expr = _scalar_field_cpp_expr(
                temperature_source,
                parameters=self.config.parameters,
                gdim=self._gdim,
            )
            blocks.append(
                textwrap.dedent(
                    f"""\
                    temperatureSource
                    {{
                        type            coded;
                        cellZone        all;
                        field           T;
                        codeAddSup
                        #{{
                            const auto& C = mesh().C();
                            const auto& V = mesh().V();
                            auto& source = eqn.source();
                            forAll(source, celli)
                            {{
                                const scalar x = C[celli].x();
                                const scalar y = C[celli].y();
                                const scalar z = C[celli].z();
                                source[celli] -= ({scalar_expr})*V[celli];
                            }}
                        #}};
                    }}
                    """
                ).rstrip()
            )

        content = _foam_dict(
            object_name="fvModels",
            location="constant",
            body="\n\n".join(blocks),
        )
        _write_text(self._current_case_dir() / "constant" / "fvModels", content)

    def _write_initial_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        temperature_entries: dict[str, str],
        cell_centres: np.ndarray,
    ) -> None:
        velocity_ic = build_vector_ic_interpolator(
            self.config.input("velocity").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="velocity",
        )
        temperature_ic = build_scalar_ic_interpolator(
            self.config.input("temperature").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="temperature",
        )
        if velocity_ic is None or temperature_ic is None:
            raise ValueError(
                "OpenFOAM-backed thermal_convection requires explicit initial "
                "condition interpolators for both velocity and temperature."
            )
        velocity_values = _pad_to_foam_vector(
            np.asarray(velocity_ic(cell_centres), dtype=float).T
        )
        temperature_values = np.asarray(temperature_ic(cell_centres), dtype=float)

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
            dimensions="[0 2 -2 0 0 0 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=pressure_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "T",
            object_name="T",
            dimensions="[0 0 0 0 0 0 0]",
            internal_values=temperature_values,
            internal_uniform=None,
            boundary_entries=temperature_entries,
        )

    def _write_placeholder_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        temperature_entries: dict[str, str],
    ) -> None:
        _write_vector_field_file(
            self._current_case_dir() / "0" / "U",
            object_name="U",
            dimensions="[0 1 -1 0 0 0 0]",
            internal_values=None,
            internal_uniform=np.zeros(3, dtype=float),
            boundary_entries=velocity_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "p",
            object_name="p",
            dimensions="[0 2 -2 0 0 0 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=pressure_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "T",
            object_name="T",
            dimensions="[0 0 0 0 0 0 0]",
            internal_values=None,
            internal_uniform=0.0,
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
        self._write_placeholder_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            temperature_entries=temperature_entries,
        )
        cell_centres = self._read_internal_cell_centres()
        self._write_initial_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            temperature_entries=temperature_entries,
            cell_centres=cell_centres,
        )
        self._write_sources()

        run_start = time.perf_counter()
        self._solve_openfoam_case()
        solve_seconds = time.perf_counter() - run_start

        num_cells, num_steps = self._sample_case_to_output(
            output=output,
            field_map={
                "pressure": "p",
                "temperature": "T",
                **{
                    f"velocity_{label}": "U"
                    for label in component_labels_for_dim(self._gdim)
                },
            },
            normalize_pressure_field="pressure",
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
