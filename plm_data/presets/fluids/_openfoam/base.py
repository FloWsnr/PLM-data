from pathlib import Path
import shlex
import shutil
import textwrap

import numpy as np

from plm_data.core.config import SimulationConfig
from plm_data.core.logging import get_logger
from plm_data.core.mesh import build_gmsh_planar_domain_model, is_gmsh_planar_domain
from plm_data.presets.base import CustomProblem

from .core import (
    _THICKNESS_2D,
    _active_openfoam_ranks,
    _foam_dict,
    _format_scalar,
    _format_vector,
    _is_supported_openfoam_mesh_domain,
    _normalise_patch_name,
    _patch_name_for_side,
    _periodic_raw_patch_name,
    _periodic_side_names,
    _run_openfoam_command,
    _set_boundary_patch_type,
    _write_text,
)
from .sampling import (
    _extract_internal_mesh,
    _openfoam_reader,
    _read_case_mesh,
    _remove_processor_dirs,
    _sample_openfoam_field_data,
    _sampled_density_array_from_pressure_temperature,
    _sampled_scalar_array,
    _sampled_vector_components,
    _uniform_points_from_bounds,
)


class _OpenFOAMProblemBase(CustomProblem):
    solver_name = ""

    def __init__(self, spec, config: SimulationConfig):
        super().__init__(spec, config)
        self._logger = get_logger("openfoam")
        self._n_subdomains = _active_openfoam_ranks()
        self._case_dir: Path | None = None
        self._openfoam_log_path: Path | None = None
        self._gdim = self.config.domain.dimension

    def _assert_supported_mesh_domain(self) -> None:
        if not _is_supported_openfoam_mesh_domain(self.config):
            raise ValueError(
                f"Preset '{self.spec.name}' does not currently support OpenFOAM "
                f"domain '{self.config.domain.type}' in {self._gdim}D."
            )

    def _current_case_dir(self) -> Path:
        if self._case_dir is None:
            raise AssertionError("OpenFOAM case directory has not been created yet.")
        return self._case_dir

    def _current_log_path(self) -> Path:
        if self._openfoam_log_path is None:
            raise AssertionError("OpenFOAM log path has not been created yet.")
        return self._openfoam_log_path

    def _run_case_command(self, command: str) -> None:
        _run_openfoam_command(
            command=command,
            cwd=self._current_case_dir(),
            log_path=self._current_log_path(),
        )

    def _prepare_case_dir(self, output_dir: Path) -> None:
        self._case_dir = output_dir / "openfoam_case"
        self._case_dir.mkdir(parents=True, exist_ok=True)
        self._openfoam_log_path = output_dir / "openfoam.log"
        self._openfoam_log_path.write_text("", encoding="utf-8")
        (self._current_case_dir() / "case.foam").touch()

    def _create_decompose_par_dict(self) -> None:
        if self._n_subdomains <= 1:
            return
        content = _foam_dict(
            object_name="decomposeParDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                numberOfSubdomains {self._n_subdomains};

                method          scotch;
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "decomposeParDict", content)

    def _write_momentum_transport(self) -> None:
        content = _foam_dict(
            object_name="momentumTransport",
            location="constant",
            body="simulationType  laminar;",
        )
        _write_text(
            self._current_case_dir() / "constant" / "momentumTransport", content
        )

    def _expected_output_times(self) -> np.ndarray:
        if self.config.time is None:
            raise ValueError(f"Preset '{self.spec.name}' requires a time section.")
        if self.config.output.num_frames > 1:
            return np.linspace(
                0.0,
                self.config.time.t_end,
                self.config.output.num_frames,
            )
        return np.array([0.0], dtype=float)

    def _write_block_mesh_rectangle(
        self, *, periodic_pairs: set[frozenset[str]]
    ) -> None:
        size_x, size_y = self.config.domain.params["size"]
        nx, ny = self.config.domain.params["mesh_resolution"]

        patch_names = {
            "x-": _patch_name_for_side(
                "x-", periodic=frozenset({"x-", "x+"}) in periodic_pairs
            ),
            "x+": _patch_name_for_side(
                "x+", periodic=frozenset({"x-", "x+"}) in periodic_pairs
            ),
            "y-": _patch_name_for_side(
                "y-", periodic=frozenset({"y-", "y+"}) in periodic_pairs
            ),
            "y+": _patch_name_for_side(
                "y+", periodic=frozenset({"y-", "y+"}) in periodic_pairs
            ),
        }
        block_mesh = _foam_dict(
            object_name="blockMeshDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                scale 1;

                vertices
                (
                    (0 0 0)
                    ({_format_scalar(size_x)} 0 0)
                    ({_format_scalar(size_x)} {_format_scalar(size_y)} 0)
                    (0 {_format_scalar(size_y)} 0)
                    (0 0 {_format_scalar(_THICKNESS_2D)})
                    ({_format_scalar(size_x)} 0 {_format_scalar(_THICKNESS_2D)})
                    ({_format_scalar(size_x)} {_format_scalar(size_y)} {_format_scalar(_THICKNESS_2D)})
                    (0 {_format_scalar(size_y)} {_format_scalar(_THICKNESS_2D)})
                );

                blocks
                (
                    hex (0 1 2 3 4 5 6 7) ({int(nx)} {int(ny)} 1) simpleGrading (1 1 1)
                );

                edges
                (
                );

                boundary
                (
                    {patch_names["x-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 3 7 4)
                        );
                    }}
                    {patch_names["x+"]}
                    {{
                        type patch;
                        faces
                        (
                            (1 5 6 2)
                        );
                    }}
                    {patch_names["y-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 4 5 1)
                        );
                    }}
                    {patch_names["y+"]}
                    {{
                        type patch;
                        faces
                        (
                            (3 2 6 7)
                        );
                    }}
                    frontAndBack
                    {{
                        type empty;
                        faces
                        (
                            (0 1 2 3)
                            (4 7 6 5)
                        );
                    }}
                );
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "blockMeshDict", block_mesh)

    def _write_block_mesh_parallelogram(
        self,
        *,
        periodic_pairs: set[frozenset[str]],
    ) -> None:
        origin = np.asarray(self.config.domain.params["origin"], dtype=float)
        axis_x = np.asarray(self.config.domain.params["axis_x"], dtype=float)
        axis_y = np.asarray(self.config.domain.params["axis_y"], dtype=float)
        nx, ny = self.config.domain.params["mesh_resolution"]
        p0 = origin
        p1 = origin + axis_x
        p2 = origin + axis_x + axis_y
        p3 = origin + axis_y
        patch_names = {
            "x-": _patch_name_for_side(
                "x-", periodic=frozenset({"x-", "x+"}) in periodic_pairs
            ),
            "x+": _patch_name_for_side(
                "x+", periodic=frozenset({"x-", "x+"}) in periodic_pairs
            ),
            "y-": _patch_name_for_side(
                "y-", periodic=frozenset({"y-", "y+"}) in periodic_pairs
            ),
            "y+": _patch_name_for_side(
                "y+", periodic=frozenset({"y-", "y+"}) in periodic_pairs
            ),
        }

        def _vertex(point: np.ndarray) -> str:
            return _format_vector(point)

        block_mesh = _foam_dict(
            object_name="blockMeshDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                scale 1;

                vertices
                (
                    {_vertex(np.array([p0[0], p0[1], 0.0]))}
                    {_vertex(np.array([p1[0], p1[1], 0.0]))}
                    {_vertex(np.array([p2[0], p2[1], 0.0]))}
                    {_vertex(np.array([p3[0], p3[1], 0.0]))}
                    {_vertex(np.array([p0[0], p0[1], _THICKNESS_2D]))}
                    {_vertex(np.array([p1[0], p1[1], _THICKNESS_2D]))}
                    {_vertex(np.array([p2[0], p2[1], _THICKNESS_2D]))}
                    {_vertex(np.array([p3[0], p3[1], _THICKNESS_2D]))}
                );

                blocks
                (
                    hex (0 1 2 3 4 5 6 7) ({int(nx)} {int(ny)} 1) simpleGrading (1 1 1)
                );

                edges
                (
                );

                boundary
                (
                    {patch_names["x-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 3 7 4)
                        );
                    }}
                    {patch_names["x+"]}
                    {{
                        type patch;
                        faces
                        (
                            (1 5 6 2)
                        );
                    }}
                    {patch_names["y-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 4 5 1)
                        );
                    }}
                    {patch_names["y+"]}
                    {{
                        type patch;
                        faces
                        (
                            (3 2 6 7)
                        );
                    }}
                    frontAndBack
                    {{
                        type empty;
                        faces
                        (
                            (0 1 2 3)
                            (4 7 6 5)
                        );
                    }}
                );
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "blockMeshDict", block_mesh)

    def _write_block_mesh_box(
        self,
        *,
        periodic_pairs: set[frozenset[str]],
    ) -> None:
        size_x, size_y, size_z = self.config.domain.params["size"]
        nx, ny, nz = self.config.domain.params["mesh_resolution"]
        periodic_x = frozenset({"x-", "x+"}) in periodic_pairs
        periodic_y = frozenset({"y-", "y+"}) in periodic_pairs
        periodic_z = frozenset({"z-", "z+"}) in periodic_pairs
        patch_names = {
            "x-": _patch_name_for_side("x-", periodic=periodic_x),
            "x+": _patch_name_for_side("x+", periodic=periodic_x),
            "y-": _patch_name_for_side("y-", periodic=periodic_y),
            "y+": _patch_name_for_side("y+", periodic=periodic_y),
            "z-": _patch_name_for_side("z-", periodic=periodic_z),
            "z+": _patch_name_for_side("z+", periodic=periodic_z),
        }
        block_mesh = _foam_dict(
            object_name="blockMeshDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                scale 1;

                vertices
                (
                    (0 0 0)
                    ({_format_scalar(size_x)} 0 0)
                    ({_format_scalar(size_x)} {_format_scalar(size_y)} 0)
                    (0 {_format_scalar(size_y)} 0)
                    (0 0 {_format_scalar(size_z)})
                    ({_format_scalar(size_x)} 0 {_format_scalar(size_z)})
                    ({_format_scalar(size_x)} {_format_scalar(size_y)} {_format_scalar(size_z)})
                    (0 {_format_scalar(size_y)} {_format_scalar(size_z)})
                );

                blocks
                (
                    hex (0 1 2 3 4 5 6 7) ({int(nx)} {int(ny)} {int(nz)}) simpleGrading (1 1 1)
                );

                edges
                (
                );

                boundary
                (
                    {patch_names["x-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 3 7 4)
                        );
                    }}
                    {patch_names["x+"]}
                    {{
                        type patch;
                        faces
                        (
                            (1 5 6 2)
                        );
                    }}
                    {patch_names["y-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 4 5 1)
                        );
                    }}
                    {patch_names["y+"]}
                    {{
                        type patch;
                        faces
                        (
                            (3 2 6 7)
                        );
                    }}
                    {patch_names["z-"]}
                    {{
                        type patch;
                        faces
                        (
                            (0 1 2 3)
                        );
                    }}
                    {patch_names["z+"]}
                    {{
                        type patch;
                        faces
                        (
                            (4 7 6 5)
                        );
                    }}
                );
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "blockMeshDict", block_mesh)

    def _write_gmsh_planar_mesh(
        self,
        *,
        periodic_pairs: dict[frozenset[str], np.ndarray],
    ) -> None:
        import gmsh

        msh_path = self._current_case_dir() / "mesh.msh"
        periodic_sides = _periodic_side_names(periodic_pairs)

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            model = gmsh.model
            model.add(self.config.domain.type)
            model.setCurrent(self.config.domain.type)
            build_gmsh_planar_domain_model(self.config.domain, model)

            boundary_curve_names: dict[int, str] = {}
            for _, physical_tag in model.getPhysicalGroups(1):
                name = model.getPhysicalName(1, physical_tag)
                if not name:
                    raise RuntimeError(
                        f"Planar Gmsh domain '{self.config.domain.type}' exposed "
                        f"an unnamed 1D physical group {physical_tag}."
                    )
                for curve_tag in model.getEntitiesForPhysicalGroup(1, physical_tag):
                    boundary_curve_names[curve_tag] = name

            surface_tags: list[int] = []
            for _, physical_tag in model.getPhysicalGroups(2):
                surface_tags.extend(model.getEntitiesForPhysicalGroup(2, physical_tag))
            if not surface_tags:
                surface_tags = [tag for dim, tag in model.getEntities(2) if dim == 2]
            if not surface_tags:
                raise RuntimeError(
                    f"Planar Gmsh domain '{self.config.domain.type}' produced no "
                    "surfaces to extrude for OpenFOAM."
                )

            extruded_entities = model.occ.extrude(
                [(2, surface_tag) for surface_tag in surface_tags],
                0.0,
                0.0,
                _THICKNESS_2D,
                numElements=[1],
                recombine=True,
            )
            model.occ.synchronize()

            volume_tags = [tag for dim, tag in extruded_entities if dim == 3]
            if not volume_tags:
                raise RuntimeError(
                    f"Failed to extrude a volume for OpenFOAM domain "
                    f"'{self.config.domain.type}'."
                )

            surface_groups = {"frontAndBack": []}
            tol = 1.0e-6
            boundary_surfaces = model.getBoundary(
                [(3, volume_tag) for volume_tag in volume_tags],
                oriented=False,
                recursive=False,
            )
            for dim, tag in boundary_surfaces:
                if dim != 2:
                    continue
                _, _, z_min, _, _, z_max = model.occ.getBoundingBox(dim, tag)
                if np.isclose(z_min, 0.0, atol=tol) and np.isclose(
                    z_max, 0.0, atol=tol
                ):
                    surface_groups["frontAndBack"].append(tag)
                    continue
                if np.isclose(z_min, _THICKNESS_2D, atol=tol) and np.isclose(
                    z_max, _THICKNESS_2D, atol=tol
                ):
                    surface_groups["frontAndBack"].append(tag)
                    continue
                curve_tags = [
                    curve_tag
                    for curve_dim, curve_tag in model.getBoundary(
                        [(2, tag)],
                        oriented=False,
                        recursive=False,
                    )
                    if curve_dim == 1 and curve_tag in boundary_curve_names
                ]
                matched_names = {
                    boundary_curve_names[curve_tag] for curve_tag in curve_tags
                }
                if len(matched_names) != 1:
                    raise RuntimeError(
                        f"OpenFOAM export for domain '{self.config.domain.type}' "
                        f"could not map surface {tag} to exactly one named "
                        f"boundary. Matched names: {sorted(matched_names)}."
                    )
                boundary_name = matched_names.pop()
                patch_name = (
                    _periodic_raw_patch_name(boundary_name)
                    if boundary_name in periodic_sides
                    else _normalise_patch_name(boundary_name)
                )
                surface_groups.setdefault(patch_name, []).append(tag)

            model.removePhysicalGroups(model.getPhysicalGroups())
            for physical_tag, (name, tags) in enumerate(
                surface_groups.items(), start=1
            ):
                if not tags:
                    raise RuntimeError(
                        f"OpenFOAM export produced no surfaces for patch '{name}' "
                        f"on domain '{self.config.domain.type}'."
                    )
                model.addPhysicalGroup(2, tags, tag=physical_tag)
                model.setPhysicalName(2, physical_tag, name)

            model.addPhysicalGroup(3, volume_tags, tag=100)
            model.setPhysicalName(3, 100, "fluid")

            model.mesh.generate(3)
            gmsh.write(str(msh_path))
        finally:
            gmsh.finalize()

    def _write_periodic_create_patch_dict(
        self,
        *,
        periodic_pairs: dict[frozenset[str], np.ndarray],
    ) -> None:
        patch_blocks = []
        for pair in sorted(periodic_pairs, key=lambda pair_key: sorted(pair_key)):
            side_a, side_b = sorted(pair, key=lambda side: _normalise_patch_name(side))
            vector_ab = periodic_pairs[pair]
            name_a = _normalise_patch_name(side_a)
            name_b = _normalise_patch_name(side_b)
            raw_a = _periodic_raw_patch_name(side_a)
            raw_b = _periodic_raw_patch_name(side_b)
            patch_blocks.append(
                textwrap.dedent(
                    f"""\
                    {name_a}
                    {{
                        patchInfo
                        {{
                            type cyclic;
                            neighbourPatch {name_b};
                            transformType translational;
                            separation {_format_vector(-vector_ab)};
                        }}
                        constructFrom patches;
                        patches ({raw_a});
                    }}
                    """
                ).rstrip()
            )
            patch_blocks.append(
                textwrap.dedent(
                    f"""\
                    {name_b}
                    {{
                        patchInfo
                        {{
                            type cyclic;
                            neighbourPatch {name_a};
                            transformType translational;
                            separation {_format_vector(vector_ab)};
                        }}
                        constructFrom patches;
                        patches ({raw_b});
                    }}
                    """
                ).rstrip()
            )
        content = _foam_dict(
            object_name="createPatchDict",
            location="system",
            body=textwrap.dedent(
                f"""\
                pointSync false;
                writeCyclicMatch false;

                patches
                {{
                {textwrap.indent("\n\n".join(patch_blocks), "    ")}
                }}
                """
            ),
        )
        _write_text(self._current_case_dir() / "system" / "createPatchDict", content)

    def _build_mesh(self, *, periodic_pairs: dict[frozenset[str], np.ndarray]) -> None:
        domain_type = self.config.domain.type
        if domain_type == "rectangle":
            self._write_block_mesh_rectangle(periodic_pairs=set(periodic_pairs))
            self._run_case_command("blockMesh")
            if periodic_pairs:
                self._write_periodic_create_patch_dict(periodic_pairs=periodic_pairs)
                self._run_case_command("createPatch -overwrite")
            return

        if domain_type == "parallelogram":
            self._write_block_mesh_parallelogram(periodic_pairs=set(periodic_pairs))
            self._run_case_command("blockMesh")
            if periodic_pairs:
                self._write_periodic_create_patch_dict(periodic_pairs=periodic_pairs)
                self._run_case_command("createPatch -overwrite")
            return

        if domain_type == "box":
            self._write_block_mesh_box(periodic_pairs=set(periodic_pairs))
            self._run_case_command("blockMesh")
            if periodic_pairs:
                self._write_periodic_create_patch_dict(periodic_pairs=periodic_pairs)
                self._run_case_command("createPatch -overwrite")
            return

        if self._gdim == 2 and is_gmsh_planar_domain(domain_type):
            self._write_gmsh_planar_mesh(periodic_pairs=periodic_pairs)
            self._run_case_command("gmshToFoam mesh.msh")
            _set_boundary_patch_type(
                self._current_case_dir() / "constant" / "polyMesh" / "boundary",
                patch_name="frontAndBack",
                patch_type="empty",
            )
            if periodic_pairs:
                self._write_periodic_create_patch_dict(periodic_pairs=periodic_pairs)
                self._run_case_command("createPatch -overwrite")
            return

        raise ValueError(
            f"Unsupported OpenFOAM mesh domain '{domain_type}' for preset "
            f"'{self.spec.name}'."
        )

    def _read_internal_cell_centres(self) -> np.ndarray:
        case_file = self._current_case_dir() / "case.foam"
        data = _read_case_mesh(case_file)
        internal_mesh = _extract_internal_mesh(data)
        centres = internal_mesh.cell_centers().points
        return np.asarray(centres[:, : self._gdim], dtype=float).T

    def _openfoam_time_values(self) -> np.ndarray:
        reader = _openfoam_reader(self._current_case_dir() / "case.foam")
        return np.asarray(reader.time_values, dtype=float)

    def _solve_openfoam_case(self) -> None:
        if self._n_subdomains <= 1:
            self._run_case_command("foamRun")
            return
        mpi_launcher = shutil.which("mpirun.openmpi") or shutil.which("mpirun")
        if mpi_launcher is None:
            raise RuntimeError("OpenFOAM parallel execution requires an MPI launcher.")
        self._run_case_command("decomposePar -force")
        self._run_case_command(
            f"{shlex.quote(mpi_launcher)} -bind-to core -map-by core -n "
            f"{self._n_subdomains} "
            "foamRun -parallel"
        )
        self._run_case_command("reconstructPar")
        _remove_processor_dirs(self._current_case_dir())

    def _sample_case_to_output(
        self,
        *,
        output,
        field_map: dict[str, str],
        normalize_pressure_field: str | None,
        scalar_offsets: dict[str, float] | None = None,
        density_from_pressure_temperature_gas_constant: float | None = None,
    ) -> tuple[int, int]:
        case_file = self._current_case_dir() / "case.foam"
        expected_times = self._expected_output_times()
        available_times = self._openfoam_time_values()
        if available_times.size == 0:
            raise RuntimeError("OpenFOAM case produced no readable output times.")

        initial_data = _read_case_mesh(case_file, time_value=float(available_times[0]))
        initial_mesh = _extract_internal_mesh(initial_data)
        resolution = tuple(self.config.output.resolution)
        sample_points, _ = _uniform_points_from_bounds(
            bounds=initial_mesh.bounds,
            resolution=resolution,
            gdim=self._gdim,
        )

        _, valid_mask = _sample_openfoam_field_data(
            internal_mesh=initial_mesh,
            sample_points=sample_points,
        )
        reshaped_valid_mask = valid_mask.reshape(resolution, order="C")
        if not np.all(reshaped_valid_mask):
            output.set_domain_mask(reshaped_valid_mask)

        num_cells = int(initial_mesh.n_cells)
        num_steps = max(
            0,
            int(round(self.config.time.t_end / self.config.time.dt))
            if self.config.time is not None
            else 0,
        )
        scalar_offsets = scalar_offsets or {}

        for expected_time in expected_times:
            time_index = int(np.argmin(np.abs(available_times - expected_time)))
            actual_time = float(available_times[time_index])
            data = _read_case_mesh(case_file, time_value=actual_time)
            internal_mesh = _extract_internal_mesh(data)
            sampled, valid_mask = _sample_openfoam_field_data(
                internal_mesh=internal_mesh,
                sample_points=sample_points,
            )
            sampled_fields: dict[str, np.ndarray] = {}
            for output_name, source_name in field_map.items():
                if output_name.startswith("velocity_"):
                    continue
                if output_name == "density" and source_name == "rho":
                    if "rho" in sampled.point_data:
                        continue
                    if (
                        density_from_pressure_temperature_gas_constant is not None
                        and "p" in sampled.point_data
                        and "T" in sampled.point_data
                    ):
                        continue
                if source_name not in sampled.point_data:
                    raise RuntimeError(
                        f"OpenFOAM field '{source_name}' was not available at "
                        f"time {actual_time}."
                    )

            if "U" in set(field_map.values()):
                components = _sampled_vector_components(
                    sampled,
                    name="U",
                    resolution=resolution,
                    valid_mask=valid_mask,
                    gdim=self._gdim,
                )
                for label, array in components.items():
                    concrete_name = f"velocity_{label}"
                    if concrete_name in output.field_names:
                        sampled_fields[concrete_name] = array

            if "p" in set(field_map.values()) and "pressure" in output.field_names:
                sampled_fields["pressure"] = _sampled_scalar_array(
                    sampled,
                    name="p",
                    resolution=resolution,
                    valid_mask=valid_mask,
                    normalize_mean=normalize_pressure_field == "pressure",
                    subtract_offset=scalar_offsets.get("pressure", 0.0),
                )

            if "T" in set(field_map.values()) and "temperature" in output.field_names:
                sampled_fields["temperature"] = _sampled_scalar_array(
                    sampled,
                    name="T",
                    resolution=resolution,
                    valid_mask=valid_mask,
                    subtract_offset=scalar_offsets.get("temperature", 0.0),
                )

            if "rho" in set(field_map.values()) and "density" in output.field_names:
                if "rho" in sampled.point_data:
                    sampled_fields["density"] = _sampled_scalar_array(
                        sampled,
                        name="rho",
                        resolution=resolution,
                        valid_mask=valid_mask,
                    )
                else:
                    if density_from_pressure_temperature_gas_constant is None:
                        raise RuntimeError(
                            "OpenFOAM density output was requested, but neither "
                            "a rho field nor a pressure/temperature fallback was "
                            "available."
                        )
                    sampled_fields["density"] = (
                        _sampled_density_array_from_pressure_temperature(
                            sampled,
                            resolution=resolution,
                            valid_mask=valid_mask,
                            gas_constant=density_from_pressure_temperature_gas_constant,
                        )
                    )

            output.write_frame(
                {}, t=float(expected_time), sampled_fields=sampled_fields
            )

        return num_cells, num_steps
