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
    _OPENFOAM_UNIVERSAL_GAS_CONSTANT,
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
    _pressure_from_density_and_patch_temperature_block,
    _pressure_from_density_temperature_block,
    _pressure_inlet_outlet_velocity_block,
    _scalar_field_cpp_expr,
    _scalar_fixed_gradient_block,
    _scalar_fixed_value_block,
    _scalar_inlet_outlet_block,
    _total_pressure_from_density_temperature_block,
    _vector_fixed_value_block,
    _write_scalar_field_file,
    _write_vector_field_file,
)


class OpenFOAMCompressibleNavierStokesProblem(_OpenFOAMProblemBase):
    solver_name = "fluid"

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._assert_supported_mesh_domain()
        if self._gdim not in {2, 3}:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D/3D, got {self._gdim}D."
            )

        self._positive_parameter("gas_constant")
        self._positive_parameter("c_v")
        self._positive_parameter("mu", allow_zero=True)
        self._positive_parameter("bulk_viscosity", allow_zero=True)
        self._positive_parameter("thermal_conductivity", allow_zero=True)
        self._scalar_degree()

        density_field = self.config.input("density")
        velocity_field = self.config.input("velocity")
        temperature_field = self.config.input("temperature")

        for context, expr in (
            ("inputs.density.initial_condition", density_field.initial_condition),
            ("inputs.velocity.initial_condition", velocity_field.initial_condition),
            (
                "inputs.temperature.initial_condition",
                temperature_field.initial_condition,
            ),
        ):
            _ensure_supported_expression(
                expr,
                allow_quadrants=True,
                allow_initial_only_types=True,
                context=context,
            )

        for context, expr in (
            ("inputs.density.source", density_field.source),
            ("inputs.velocity.source", velocity_field.source),
            ("inputs.temperature.source", temperature_field.source),
        ):
            if expr is None:
                continue
            _ensure_supported_expression(
                expr,
                allow_quadrants=False,
                context=context,
            )

        for field_name, expr in (
            ("density", density_field.initial_condition),
            ("velocity", velocity_field.initial_condition),
            ("temperature", temperature_field.initial_condition),
        ):
            if expr.type == "custom":
                raise ValueError(
                    "OpenFOAM-backed compressible_navier_stokes does not support "
                    f"{field_name}.initial_condition.type='custom'."
                )

    def _positive_parameter(self, name: str, *, allow_zero: bool = False) -> float:
        value = float(self.config.parameters[name])
        if allow_zero:
            if value < 0.0:
                raise ValueError(
                    f"Preset '{self.spec.name}' requires parameter '{name}' >= 0. "
                    f"Got {value}."
                )
        elif value <= 0.0:
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
                "Preset 'compressible_navier_stokes' requires integer parameter "
                f"'k' >= 1. Got {raw_degree}."
            )
        return degree

    def _gas_constant(self) -> float:
        return self._positive_parameter("gas_constant")

    def _heat_capacity_cv(self) -> float:
        return self._positive_parameter("c_v")

    def _dynamic_viscosity(self) -> float:
        return self._positive_parameter("mu", allow_zero=True)

    def _bulk_viscosity(self) -> float:
        return self._positive_parameter("bulk_viscosity", allow_zero=True)

    def _thermal_conductivity(self) -> float:
        return self._positive_parameter("thermal_conductivity", allow_zero=True)

    def _molecular_weight(self) -> float:
        return _OPENFOAM_UNIVERSAL_GAS_CONSTANT / self._gas_constant()

    def _bulk_viscosity_correction(self) -> float:
        return self._bulk_viscosity() + (2.0 / 3.0) * self._dynamic_viscosity()

    def _channel_open_boundary_role(self, side: str) -> str:
        if self.config.domain.type in {
            "channel_obstacle",
            "y_bifurcation",
            "venturi_channel",
            "porous_channel",
            "serpentine_channel",
            "airfoil_channel",
            "side_cavity_channel",
        }:
            if side.startswith("inlet"):
                return "inlet"
            if side.startswith("outlet"):
                return "outlet"
        return "generic"

    def _density_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("density")

    def _velocity_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("velocity")

    def _temperature_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("temperature")

    def _periodic_vectors(self) -> dict[frozenset[str], np.ndarray]:
        density_pairs = self._density_boundary().periodic_pair_keys()
        velocity_pairs = self._velocity_boundary().periodic_pair_keys()
        temperature_pairs = self._temperature_boundary().periodic_pair_keys()
        if density_pairs != velocity_pairs or density_pairs != temperature_pairs:
            raise ValueError(
                "Density, velocity, and temperature boundary conditions must use "
                "identical periodic side pairs."
            )
        return _resolve_periodic_vectors(self.config, density_pairs)

    def _validate_boundary_conditions(self) -> None:
        expected = _expected_boundary_names(self.config)
        for field_name, boundary_field in (
            ("density", self._density_boundary()),
            ("velocity", self._velocity_boundary()),
            ("temperature", self._temperature_boundary()),
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
            temperature_entry = _single_boundary_entry(
                boundary_field=self._temperature_boundary(),
                side=side,
            )

            if density_entry.type not in {"dirichlet", "periodic"}:
                raise ValueError(
                    "OpenFOAM-backed compressible_navier_stokes only supports "
                    f"density dirichlet/periodic boundaries. Side '{side}' uses "
                    f"'{density_entry.type}'."
                )
            if velocity_entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    "OpenFOAM-backed compressible_navier_stokes only supports "
                    f"velocity dirichlet, zero-style neumann, or periodic "
                    f"boundaries. Side '{side}' uses '{velocity_entry.type}'."
                )
            if temperature_entry.type not in {"dirichlet", "neumann", "periodic"}:
                raise ValueError(
                    "OpenFOAM-backed compressible_navier_stokes only supports "
                    f"temperature dirichlet/neumann/periodic boundaries. Side "
                    f"'{side}' uses '{temperature_entry.type}'."
                )

            if density_entry.value is not None:
                _ensure_supported_expression(
                    density_entry.value,
                    allow_quadrants=False,
                    context=f"boundary_conditions.density.{side}",
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

            if (
                temperature_entry.type == "neumann"
                and temperature_entry.value is not None
                and not is_exact_zero_field_expression(
                    temperature_entry.value,
                    self.config.parameters,
                )
                and self._thermal_conductivity() <= 0.0
            ):
                raise ValueError(
                    "OpenFOAM-backed compressible_navier_stokes requires "
                    "thermal_conductivity > 0 when non-zero temperature Neumann "
                    f"data is used. Side '{side}' is invalid."
                )

            if velocity_entry.type == "neumann" and velocity_entry.value is not None:
                if not is_exact_zero_field_expression(
                    velocity_entry.value,
                    self.config.parameters,
                ):
                    raise ValueError(
                        "OpenFOAM-backed compressible_navier_stokes currently only "
                        "supports zero traction / zero-gradient style velocity "
                        f"neumann conditions. Side '{side}' is non-zero."
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

                adjustTimeStep  yes;
                maxCo           0.5;
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
                        mu              {_format_scalar(self._dynamic_viscosity())};
                        kappa           {_format_scalar(self._thermal_conductivity())};
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
                    div(phi,U)              Gauss upwind;
                    div(phi,e)              Gauss upwind;
                    div(phi,K)              Gauss linear;
                    div(phi,(p|rho))        Gauss linear;
                    div(U)                  Gauss linear;
                    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
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
                    "rho"
                    {
                        solver          diagonal;
                    }

                    "rhoFinal"
                    {
                        $rho;
                    }

                    "p"
                    {
                        solver          PCG;
                        preconditioner  DIC;
                        tolerance       1e-8;
                        relTol          0.05;
                    }

                    "pFinal"
                    {
                        $p;
                        relTol          0;
                    }

                    "(U|e)"
                    {
                        solver          PBiCGStab;
                        preconditioner  DILU;
                        tolerance       1e-9;
                        relTol          0;
                    }

                    "(U|e)Final"
                    {
                        $U;
                        relTol          0;
                    }
                }

                PIMPLE
                {
                    momentumPredictor          yes;
                    nOuterCorrectors           2;
                    nCorrectors                2;
                    nNonOrthogonalCorrectors   0;
                    pRefCell                   0;
                    pRefValue                  0;
                }

                relaxationFactors
                {
                    fields
                    {
                        p       0.3;
                        rho     1.0;
                    }

                    equations
                    {
                        U       0.7;
                        e       0.7;
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
        density_boundary = self._density_boundary()
        velocity_boundary = self._velocity_boundary()
        temperature_boundary = self._temperature_boundary()
        gas_constant = self._gas_constant()
        thermal_conductivity = self._thermal_conductivity()

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
            temperature_entry = _single_boundary_entry(
                boundary_field=temperature_boundary,
                side=side,
            )
            patch_name = _periodic_patch_name(side, periodic_pairs)
            boundary_role = self._channel_open_boundary_role(side)

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

            assert density_entry.value is not None

            if velocity_entry.type == "dirichlet":
                assert velocity_entry.value is not None
                if boundary_role == "outlet":
                    velocity_entries[patch_name] = (
                        _pressure_inlet_outlet_velocity_block(
                            expr=velocity_entry.value,
                            parameters=self.config.parameters,
                            gdim=self._gdim,
                        )
                    )
                else:
                    velocity_entries[patch_name] = _vector_fixed_value_block(
                        expr=velocity_entry.value,
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"U_{patch_name}",
                        zero_patch_type="noSlip",
                    )
            elif velocity_entry.type == "neumann":
                velocity_entries[patch_name] = textwrap.dedent(
                    """\
                    type            pressureInletOutletVelocity;
                    value           uniform (0 0 0);
                    """
                )
            else:
                raise AssertionError(
                    f"Unhandled velocity boundary type '{velocity_entry.type}'."
                )

            if temperature_entry.type == "dirichlet":
                assert temperature_entry.value is not None
                if boundary_role == "outlet":
                    temperature_entries[patch_name] = _scalar_inlet_outlet_block(
                        expr=temperature_entry.value,
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"T_{patch_name}",
                    )
                    pressure_entries[patch_name] = (
                        _total_pressure_from_density_temperature_block(
                            density_expr=density_entry.value,
                            temperature_expr=temperature_entry.value,
                            gas_constant=gas_constant,
                            parameters=self.config.parameters,
                            gdim=self._gdim,
                            field_name=f"p_{patch_name}",
                        )
                    )
                elif boundary_role == "inlet":
                    temperature_entries[patch_name] = _scalar_fixed_value_block(
                        expr=temperature_entry.value,
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"T_{patch_name}",
                    )
                    pressure_entries[patch_name] = "type            zeroGradient;"
                else:
                    temperature_entries[patch_name] = _scalar_fixed_value_block(
                        expr=temperature_entry.value,
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"T_{patch_name}",
                    )
                    pressure_entries[patch_name] = (
                        _pressure_from_density_temperature_block(
                            density_expr=density_entry.value,
                            temperature_expr=temperature_entry.value,
                            gas_constant=gas_constant,
                            parameters=self.config.parameters,
                            gdim=self._gdim,
                            field_name=f"p_{patch_name}",
                        )
                    )
                continue

            if temperature_entry.type == "neumann":
                assert temperature_entry.value is not None
                gradient_scale = (
                    1.0 / thermal_conductivity if thermal_conductivity > 0.0 else 1.0
                )
                temperature_entries[patch_name] = _scalar_fixed_gradient_block(
                    expr=temperature_entry.value,
                    parameters=self.config.parameters,
                    gdim=self._gdim,
                    field_name=f"T_{patch_name}",
                    gradient_scale=gradient_scale,
                )
                if boundary_role == "inlet":
                    pressure_entries[patch_name] = "type            zeroGradient;"
                else:
                    pressure_entries[patch_name] = (
                        _pressure_from_density_and_patch_temperature_block(
                            density_expr=density_entry.value,
                            gas_constant=gas_constant,
                            parameters=self.config.parameters,
                            gdim=self._gdim,
                            field_name=f"p_{patch_name}",
                        )
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

        density_source = self.config.input("density").source
        if density_source is not None and not is_exact_zero_field_expression(
            density_source,
            self.config.parameters,
        ):
            scalar_expr = _scalar_field_cpp_expr(
                density_source,
                parameters=self.config.parameters,
                gdim=self._gdim,
            )
            blocks.append(
                textwrap.dedent(
                    f"""\
                    densitySource
                    {{
                        type            coded;
                        cellZone        all;
                        field           rho;
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
                        codeAddRhoSup
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
                        field           e;
                        codeAddRhoSup
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

        bulk_correction = self._bulk_viscosity_correction()
        if not np.isclose(bulk_correction, 0.0):
            blocks.append(
                textwrap.dedent(
                    f"""\
                    bulkViscosityMomentumCorrection
                    {{
                        type            coded;
                        cellZone        all;
                        field           U;
                        codeInclude
                        #{{
                            #include "fvcDiv.H"
                            #include "fvcGrad.H"
                        #}};
                        codeAddRhoSup
                        #{{
                            const auto& V = mesh().V();
                            const auto tDivU = fvc::div
                            (
                                mesh().lookupObject<volVectorField>("U")
                            );
                            const auto& divU = tDivU();
                            const auto tGradDivU = fvc::grad(divU);
                            const auto& gradDivU = tGradDivU();
                            auto& source = eqn.source();
                            forAll(source, celli)
                            {{
                                source[celli] -=
                                    {_format_scalar(bulk_correction)}
                                   *gradDivU[celli]*V[celli];
                            }}
                        #}};
                    }}
                    """
                ).rstrip()
            )
            blocks.append(
                textwrap.dedent(
                    f"""\
                    bulkViscosityHeatingCorrection
                    {{
                        type            coded;
                        cellZone        all;
                        field           e;
                        codeInclude
                        #{{
                            #include "fvcDiv.H"
                        #}};
                        codeAddRhoSup
                        #{{
                            const auto& V = mesh().V();
                            const auto tDivU = fvc::div
                            (
                                mesh().lookupObject<volVectorField>("U")
                            );
                            const auto& divU = tDivU();
                            auto& source = eqn.source();
                            forAll(source, celli)
                            {{
                                source[celli] -=
                                    {_format_scalar(bulk_correction)}
                                   *Foam::sqr(divU[celli])*V[celli];
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
        temperature_ic = build_scalar_ic_interpolator(
            self.config.input("temperature").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="temperature",
        )
        if density_ic is None or velocity_ic is None or temperature_ic is None:
            raise ValueError(
                "OpenFOAM-backed compressible_navier_stokes requires explicit "
                "initial condition interpolators for density, velocity, and "
                "temperature."
            )

        density_values = np.asarray(density_ic(cell_centres), dtype=float)
        velocity_values = _pad_to_foam_vector(
            np.asarray(velocity_ic(cell_centres), dtype=float).T
        )
        temperature_values = np.asarray(temperature_ic(cell_centres), dtype=float)
        pressure_values = density_values * self._gas_constant() * temperature_values

        if np.min(density_values) <= 0.0:
            raise ValueError(
                "OpenFOAM-backed compressible_navier_stokes requires strictly "
                "positive initial density."
            )
        if np.min(temperature_values) <= 0.0:
            raise ValueError(
                "OpenFOAM-backed compressible_navier_stokes requires strictly "
                "positive initial temperature."
            )
        if np.min(pressure_values) <= 0.0:
            raise ValueError(
                "OpenFOAM-backed compressible_navier_stokes requires strictly "
                "positive initial pressure."
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
            dimensions="[1 -1 -2 0 0 0 0]",
            internal_values=None,
            internal_uniform=max(self._gas_constant(), 1.0),
            boundary_entries=pressure_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "T",
            object_name="T",
            dimensions="[0 0 0 1 0 0 0]",
            internal_values=None,
            internal_uniform=1.0,
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
