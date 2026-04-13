import textwrap
import time

import numpy as np

from plm_data.core.config import BoundaryFieldConfig, SimulationConfig
from plm_data.core.initial_conditions import build_vector_ic_interpolator
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
    _write_text,
)
from .fields import (
    _vector_fixed_value_block,
    _write_scalar_field_file,
    _write_vector_field_file,
)


class OpenFOAMMHDProblem(_OpenFOAMProblemBase):
    solver_name = "mhdFoam"
    solver_executable = "mhdFoam"

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._assert_supported_mesh_domain()
        if self._gdim not in {2, 3}:
            raise ValueError(
                f"Preset '{self.spec.name}' only supports 2D/3D, got {self._gdim}D."
            )

        for field_name in ("velocity", "magnetic_field"):
            field = self.config.input(field_name)
            _ensure_supported_expression(
                field.initial_condition,
                allow_quadrants=True,
                allow_initial_only_types=True,
                context=f"inputs.{field_name}.initial_condition",
            )
            if field.initial_condition.type == "custom":
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} does not support "
                    f"{field_name}.initial_condition.type='custom'."
                )
            if field.source is not None and not is_exact_zero_field_expression(
                field.source,
                self.config.parameters,
            ):
                raise ValueError(
                    f"OpenFOAM-backed {self.spec.name} does not support non-zero "
                    f"sources for '{field_name}'."
                )

        reynolds = float(self.config.parameters["Re"])
        magnetic_reynolds = float(self.config.parameters["Rm"])
        if reynolds <= 0.0:
            raise ValueError(
                f"Preset '{self.spec.name}' requires parameter 'Re' > 0. Got "
                f"{reynolds}."
            )
        if magnetic_reynolds <= 0.0:
            raise ValueError(
                f"Preset '{self.spec.name}' requires parameter 'Rm' > 0. Got "
                f"{magnetic_reynolds}."
            )

    def _velocity_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("velocity")

    def _magnetic_boundary(self) -> BoundaryFieldConfig:
        return self.config.boundary_field("magnetic_field")

    def _periodic_vectors(self) -> dict[frozenset[str], np.ndarray]:
        velocity_pairs = self._velocity_boundary().periodic_pair_keys()
        magnetic_pairs = self._magnetic_boundary().periodic_pair_keys()
        if velocity_pairs != magnetic_pairs:
            raise ValueError(
                "Velocity and magnetic-field boundary conditions must use identical "
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
            boundary_field=self._magnetic_boundary(),
            expected_sides=expected,
            preset_name=self.spec.name,
            field_name="magnetic_field",
        )
        if (
            self._velocity_boundary().periodic_pair_keys()
            != self._magnetic_boundary().periodic_pair_keys()
        ):
            raise ValueError(
                "Velocity and magnetic-field boundary conditions must use identical "
                "periodic side pairs."
            )
        if all(
            _single_boundary_entry(
                boundary_field=self._magnetic_boundary(),
                side=side,
            ).type
            == "periodic"
            for side in expected
        ):
            raise ValueError(
                "OpenFOAM-backed mhdFoam requires at least one non-periodic "
                "magnetic-field boundary so pB can be anchored; fully periodic "
                "magnetic domains are not supported."
            )

        for field_name, boundary_field in (
            ("velocity", self._velocity_boundary()),
            ("magnetic_field", self._magnetic_boundary()),
        ):
            for side in expected:
                entry = _single_boundary_entry(
                    boundary_field=boundary_field,
                    side=side,
                )
                if entry.type not in {"dirichlet", "neumann", "periodic"}:
                    raise ValueError(
                        f"OpenFOAM-backed {self.spec.name} does not support "
                        f"{field_name} boundary operator '{entry.type}' on side "
                        f"'{side}'."
                    )
                if entry.value is not None:
                    _ensure_supported_expression(
                        entry.value,
                        allow_quadrants=False,
                        context=f"boundary_conditions.{field_name}.{side}",
                    )
                if entry.type == "neumann" and (
                    entry.value is None
                    or not is_exact_zero_field_expression(
                        entry.value,
                        self.config.parameters,
                    )
                ):
                    raise ValueError(
                        f"OpenFOAM-backed {self.spec.name} currently only supports "
                        f"zero-gradient style neumann conditions for '{field_name}' "
                        f"on side '{side}'."
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
        magnetic_reynolds = float(self.config.parameters["Rm"])
        content = _foam_dict(
            object_name="physicalProperties",
            location="constant",
            body=textwrap.dedent(
                f"""\
                rho             [1 -3 0 0 0 0 0] 1;
                nu              [0 2 -1 0 0 0 0] {_format_scalar(1.0 / reynolds)};
                mu              [1 1 -2 0 0 -2 0] 1;
                sigma           [-1 -3 3 0 0 2 0] {_format_scalar(magnetic_reynolds)};
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
                    div(phi,U)      Gauss linear;
                    div(phiB,U)     Gauss linear;
                    div(phi,B)      Gauss linear;
                    div(phiB,((2*DBU)*B)) Gauss linear;
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
                        solver          PCG;
                        preconditioner  DIC;
                        tolerance       1e-6;
                        relTol          0.05;
                    }

                    pFinal
                    {
                        $p;
                        relTol          0;
                    }

                    pB
                    {
                        $p;
                        tolerance       1e-5;
                        relTol          0;
                    }

                    pBFinal
                    {
                        $pB;
                        relTol          0;
                    }

                    "(U|B).*"
                    {
                        solver          smoothSolver;
                        smoother        symGaussSeidel;
                        tolerance       1e-5;
                        relTol          0;
                    }
                }

                PISO
                {
                    nCorrectors                 3;
                    nNonOrthogonalCorrectors    0;
                    pRefCell                    0;
                    pRefValue                   0;
                }

                BPISO
                {
                    nCorrectors     3;
                }
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "fvSolution", content)

    def _field_patch_entries(
        self,
        *,
        periodic_pairs: dict[frozenset[str], np.ndarray],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
        velocity_entries: dict[str, str] = {}
        pressure_entries: dict[str, str] = {}
        magnetic_entries: dict[str, str] = {}
        magnetic_pressure_entries: dict[str, str] = {}
        zero_scalar_fixed_value = textwrap.dedent(
            """\
            type            fixedValue;
            value           uniform 0;
            """
        )

        field_specs = (
            (
                self._velocity_boundary(),
                velocity_entries,
                pressure_entries,
                "U",
                True,
            ),
            (
                self._magnetic_boundary(),
                magnetic_entries,
                magnetic_pressure_entries,
                "B",
                False,
            ),
        )

        for (
            boundary_field,
            vector_entries,
            scalar_entries,
            vector_field_name,
            zero_is_noslip,
        ) in field_specs:
            for side in boundary_field.sides:
                entry = _single_boundary_entry(
                    boundary_field=boundary_field,
                    side=side,
                )
                patch_name = _periodic_patch_name(side, periodic_pairs)
                if entry.type == "periodic":
                    cyclic_entry = "type            cyclic;"
                    vector_entries[_normalise_patch_name(side)] = cyclic_entry
                    scalar_entries[_normalise_patch_name(side)] = cyclic_entry
                    continue

                assert entry.value is not None
                if entry.type == "dirichlet":
                    vector_entries[patch_name] = _vector_fixed_value_block(
                        expr=entry.value,
                        parameters=self.config.parameters,
                        gdim=self._gdim,
                        field_name=f"{vector_field_name}_{patch_name}",
                        zero_patch_type="noSlip" if zero_is_noslip else None,
                    )
                    scalar_entries[patch_name] = "type            zeroGradient;"
                    continue

                vector_entries[patch_name] = "type            zeroGradient;"
                scalar_entries[patch_name] = zero_scalar_fixed_value

        if not any("fixedValue" in body for body in magnetic_pressure_entries.values()):
            for patch_name, body in magnetic_pressure_entries.items():
                if "cyclic" in body or "empty" in body:
                    continue
                magnetic_pressure_entries[patch_name] = textwrap.dedent(
                    """\
                    type            fixedValue;
                    value           uniform 0;
                    """
                )
                break

        if self._gdim == 2:
            for entries in (
                velocity_entries,
                pressure_entries,
                magnetic_entries,
                magnetic_pressure_entries,
            ):
                entries["frontAndBack"] = "type            empty;"
        return (
            velocity_entries,
            pressure_entries,
            magnetic_entries,
            magnetic_pressure_entries,
        )

    def _write_initial_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        magnetic_entries: dict[str, str],
        magnetic_pressure_entries: dict[str, str],
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
                f"OpenFOAM-backed {self.spec.name} requires an explicit velocity "
                "initial-condition interpolator."
            )
        magnetic_ic = build_vector_ic_interpolator(
            self.config.input("magnetic_field").initial_condition,
            self.config.parameters,
            gdim=self._gdim,
            seed=self.config.seed,
            stream_id="magnetic_field",
        )
        if magnetic_ic is None:
            raise ValueError(
                f"OpenFOAM-backed {self.spec.name} requires an explicit magnetic "
                "initial-condition interpolator."
            )

        velocity_values = _pad_to_foam_vector(
            np.asarray(velocity_ic(cell_centres), dtype=float).T
        )
        magnetic_values = _pad_to_foam_vector(
            np.asarray(magnetic_ic(cell_centres), dtype=float).T
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
        _write_vector_field_file(
            self._current_case_dir() / "0" / "B",
            object_name="B",
            dimensions="[1 0 -2 0 0 -1 0]",
            internal_values=magnetic_values,
            internal_uniform=None,
            boundary_entries=magnetic_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "pB",
            object_name="pB",
            dimensions="[1 1 -3 0 0 -1 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=magnetic_pressure_entries,
        )

    def _write_placeholder_fields(
        self,
        *,
        velocity_entries: dict[str, str],
        pressure_entries: dict[str, str],
        magnetic_entries: dict[str, str],
        magnetic_pressure_entries: dict[str, str],
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
        _write_vector_field_file(
            self._current_case_dir() / "0" / "B",
            object_name="B",
            dimensions="[1 0 -2 0 0 -1 0]",
            internal_values=None,
            internal_uniform=np.zeros(3, dtype=float),
            boundary_entries=magnetic_entries,
        )
        _write_scalar_field_file(
            self._current_case_dir() / "0" / "pB",
            object_name="pB",
            dimensions="[1 1 -3 0 0 -1 0]",
            internal_values=None,
            internal_uniform=0.0,
            boundary_entries=magnetic_pressure_entries,
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
        self._write_fv_schemes()
        self._write_fv_solution()
        self._create_decompose_par_dict()
        self._build_mesh(periodic_pairs=periodic_pairs)
        (
            velocity_entries,
            pressure_entries,
            magnetic_entries,
            magnetic_pressure_entries,
        ) = self._field_patch_entries(periodic_pairs=periodic_pairs)
        self._write_placeholder_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            magnetic_entries=magnetic_entries,
            magnetic_pressure_entries=magnetic_pressure_entries,
        )
        cell_centres = self._read_internal_cell_centres()
        self._write_initial_fields(
            velocity_entries=velocity_entries,
            pressure_entries=pressure_entries,
            magnetic_entries=magnetic_entries,
            magnetic_pressure_entries=magnetic_pressure_entries,
            cell_centres=cell_centres,
        )

        run_start = time.perf_counter()
        self._solve_openfoam_case()
        solve_seconds = time.perf_counter() - run_start

        num_cells, num_steps = self._sample_case_to_output(
            output=output,
            field_map={
                "pressure": "p",
                "magnetic_constraint": "pB",
                **{
                    f"velocity_{label}": "U"
                    for label in component_labels_for_dim(self._gdim)
                },
                **{
                    f"magnetic_field_{label}": "B"
                    for label in component_labels_for_dim(self._gdim)
                },
            },
            normalize_pressure_field="pressure",
        )
        num_dofs = num_cells * (2 * self._gdim + 2)
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
