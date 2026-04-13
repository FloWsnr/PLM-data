import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import textwrap

import numpy as np

from plm_data.core.config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
    FieldExpressionConfig,
    SimulationConfig,
)
from plm_data.core.mesh import is_gmsh_planar_domain
from plm_data.core.spatial_fields import (
    component_expressions,
    component_labels_for_dim,
    resolve_param_ref,
)

_THICKNESS_2D = 1.0
_FOAM_BASHRC = "/opt/openfoam13/etc/bashrc"
_PI_LITERAL = repr(math.pi)
_OPENFOAM_UNIVERSAL_GAS_CONSTANT = 8314.46261815324
_SIDE_TO_PATCH = {
    "x-": "x_minus",
    "x+": "x_plus",
    "y-": "y_minus",
    "y+": "y_plus",
    "z-": "z_minus",
    "z+": "z_plus",
}


def _format_scalar(value: float) -> str:
    return f"{float(value):.16g}"


def _format_vector(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return "(" + " ".join(_format_scalar(float(v)) for v in values) + ")"


def _pad_to_foam_vector(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.shape[0] >= 3:
            return array[:3]
        return np.pad(array, (0, 3 - array.shape[0]))
    if array.ndim == 2:
        if array.shape[1] >= 3:
            return array[:, :3]
        return np.pad(array, ((0, 0), (0, 3 - array.shape[1])))
    raise ValueError(f"Expected a 1D/2D vector array. Got shape {array.shape}.")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _set_boundary_patch_type(
    boundary_file: Path,
    *,
    patch_name: str,
    patch_type: str,
) -> None:
    content = boundary_file.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(^\s*{re.escape(patch_name)}\s*\{{.*?^\s*type\s+)(\w+)(\s*;)",
        flags=re.MULTILINE | re.DOTALL,
    )
    updated, count = pattern.subn(rf"\1{patch_type}\3", content, count=1)
    if count != 1:
        raise RuntimeError(
            f"Failed to update patch '{patch_name}' to type '{patch_type}' in "
            f"{boundary_file}."
        )
    boundary_file.write_text(updated, encoding="utf-8")


def _write_foam_header(
    *,
    class_name: str,
    object_name: str,
    location: str | None = None,
) -> str:
    location_line = ""
    if location is not None:
        location_line = f'    location    "{location}";\n'
    return textwrap.dedent(
        f"""\
        /*--------------------------------*- C++ -*----------------------------------*\\
          =========                 |
          \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
           \\\\    /   O peration     | Website:  https://openfoam.org
            \\\\  /    A nd           | Version:  13
             \\\\/     M anipulation  |
        \\*---------------------------------------------------------------------------*/
        FoamFile
        {{
            format      ascii;
            class       {class_name};
        {location_line}    object      {object_name};
        }}
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

        """
    )


def _foam_dict(
    *,
    object_name: str,
    body: str,
    location: str | None = None,
) -> str:
    return (
        _write_foam_header(
            class_name="dictionary",
            object_name=object_name,
            location=location,
        )
        + body.strip()
        + "\n\n// ************************************************************************* //\n"
    )


def _normalise_patch_name(name: str) -> str:
    if name in _SIDE_TO_PATCH:
        return _SIDE_TO_PATCH[name]
    return name.replace("-", "_minus_").replace("+", "_plus_")


def _periodic_raw_patch_name(side: str) -> str:
    return f"{_normalise_patch_name(side)}_raw"


def _patch_name_for_side(side: str, *, periodic: bool) -> str:
    return _periodic_raw_patch_name(side) if periodic else _normalise_patch_name(side)


def _active_openfoam_ranks() -> int:
    raw = os.environ.get("PLM_OPENFOAM_NPROCS")
    if raw is None:
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError(f"PLM_OPENFOAM_NPROCS must be at least 1, got {value}.")
    return value


def _run_openfoam_command(
    *,
    command: str,
    cwd: Path,
    log_path: Path,
) -> None:
    if not Path(_FOAM_BASHRC).is_file():
        raise RuntimeError(
            f"OpenFOAM 13 setup script was not found at '{_FOAM_BASHRC}'."
        )

    shell_command = f". {shlex.quote(_FOAM_BASHRC)} >/dev/null 2>&1 && {command}"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {command}\n")
        log_file.flush()
        completed = subprocess.run(
            ["/bin/bash", "-lc", shell_command],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"OpenFOAM command failed with exit code {completed.returncode}: "
            f"{command}. See {log_path}."
        )


def _single_boundary_entry(
    *,
    boundary_field: BoundaryFieldConfig,
    side: str,
) -> BoundaryConditionConfig:
    entries = boundary_field.side_conditions(side)
    if len(entries) != 1:
        raise ValueError(
            f"OpenFOAM backend requires exactly one boundary operator on side "
            f"'{side}'. Got {len(entries)}."
        )
    return entries[0]


def _supported_standard_sides(gdim: int) -> set[str]:
    if gdim == 2:
        return {"x-", "x+", "y-", "y+"}
    return {"x-", "x+", "y-", "y+", "z-", "z+"}


def _validate_side_names(
    *,
    boundary_field: BoundaryFieldConfig,
    expected_sides: set[str],
    preset_name: str,
    field_name: str,
) -> None:
    actual_sides = set(boundary_field.sides)
    if actual_sides != expected_sides:
        raise ValueError(
            f"Preset '{preset_name}' requires boundary field '{field_name}' to use "
            f"exactly sides {sorted(expected_sides)}. Got {sorted(actual_sides)}."
        )


def _expected_boundary_names(config: SimulationConfig) -> set[str]:
    domain_type = config.domain.type
    gdim = config.domain.dimension
    if domain_type in {"rectangle", "box", "parallelogram"}:
        return _supported_standard_sides(gdim)
    if domain_type in {"disk", "dumbbell"}:
        return {"outer"}
    if domain_type == "l_shape":
        return {"outer", "notch"}
    if domain_type == "channel_obstacle":
        return {"inlet", "outlet", "walls", "obstacle"}
    if domain_type == "annulus":
        return {"inner", "outer"}
    if domain_type == "y_bifurcation":
        return {"inlet", "outlet_upper", "outlet_lower", "walls"}
    if domain_type == "venturi_channel":
        return {"inlet", "outlet", "walls"}
    if domain_type == "porous_channel":
        return {"inlet", "outlet", "walls", "obstacles"}
    if domain_type == "serpentine_channel":
        return {"inlet", "outlet", "walls"}
    if domain_type == "airfoil_channel":
        return {"inlet", "outlet", "walls", "airfoil"}
    if domain_type == "side_cavity_channel":
        return {"inlet", "outlet", "walls"}
    if domain_type == "multi_hole_plate":
        names = {"outer"}
        for hole_raw in config.domain.params["holes"]:
            if not isinstance(hole_raw, dict):
                raise ValueError(
                    "multi_hole_plate holes must be mappings after realization."
                )
            names.add(str(hole_raw.get("boundary_name", "holes")))
        return names
    raise ValueError(
        f"OpenFOAM backend does not know the boundary layout for domain "
        f"'{domain_type}'."
    )


def _is_supported_openfoam_mesh_domain(config: SimulationConfig) -> bool:
    domain_type = config.domain.type
    if domain_type in {"rectangle", "parallelogram"}:
        return config.domain.dimension == 2
    if domain_type == "box":
        return config.domain.dimension == 3
    return config.domain.dimension == 2 and is_gmsh_planar_domain(domain_type)


def _periodic_side_names(
    periodic_pairs: dict[frozenset[str], np.ndarray] | set[frozenset[str]],
) -> set[str]:
    names: set[str] = set()
    for pair in periodic_pairs:
        names.update(pair)
    return names


def _periodic_patch_name(
    side: str,
    periodic_pairs: dict[frozenset[str], np.ndarray] | set[frozenset[str]],
) -> str:
    return _patch_name_for_side(
        side, periodic=side in _periodic_side_names(periodic_pairs)
    )


def _builtin_translation_periodic_vectors(
    config: SimulationConfig,
) -> dict[frozenset[str], np.ndarray]:
    params = config.domain.params
    if config.domain.type == "rectangle":
        size = np.asarray(params["size"], dtype=float)
        vectors = {
            frozenset({"x-", "x+"}): np.array([size[0], 0.0, 0.0], dtype=float),
            frozenset({"y-", "y+"}): np.array([0.0, size[1], 0.0], dtype=float),
        }
        if config.domain.dimension == 3:
            vectors[frozenset({"z-", "z+"})] = np.array(
                [0.0, 0.0, size[2]],
                dtype=float,
            )
        return vectors
    if config.domain.type == "box":
        size = np.asarray(params["size"], dtype=float)
        return {
            frozenset({"x-", "x+"}): np.array([size[0], 0.0, 0.0], dtype=float),
            frozenset({"y-", "y+"}): np.array([0.0, size[1], 0.0], dtype=float),
            frozenset({"z-", "z+"}): np.array([0.0, 0.0, size[2]], dtype=float),
        }
    if config.domain.type == "parallelogram":
        axis_x = np.asarray(params["axis_x"], dtype=float)
        axis_y = np.asarray(params["axis_y"], dtype=float)
        return {
            frozenset({"x-", "x+"}): np.array([axis_x[0], axis_x[1], 0.0], dtype=float),
            frozenset({"y-", "y+"}): np.array([axis_y[0], axis_y[1], 0.0], dtype=float),
        }
    return {}


def _config_translation_periodic_vectors(
    config: SimulationConfig,
) -> dict[frozenset[str], np.ndarray]:
    vectors: dict[frozenset[str], np.ndarray] = {}
    identity = np.eye(config.domain.dimension, dtype=float)
    for name, map_config in config.domain.periodic_maps.items():
        matrix = np.asarray(map_config.matrix, dtype=float)
        if not np.allclose(matrix, identity):
            raise ValueError(
                f"OpenFOAM periodic map '{name}' on domain '{config.domain.type}' "
                "must be translational (identity transform matrix)."
            )
        offset = _pad_to_foam_vector(np.asarray(map_config.offset, dtype=float))
        pair = frozenset({map_config.slave, map_config.master})
        side_a, side_b = sorted(pair, key=_normalise_patch_name)
        if side_a == map_config.slave and side_b == map_config.master:
            vector_ab = offset
        elif side_a == map_config.master and side_b == map_config.slave:
            vector_ab = -offset
        else:
            raise AssertionError(
                f"Unexpected periodic side ordering for pair {sorted(pair)}."
            )
        if pair in vectors:
            raise ValueError(
                f"Duplicate OpenFOAM periodic map for pair {sorted(pair)}."
            )
        vectors[pair] = vector_ab
    return vectors


def _resolve_periodic_vectors(
    config: SimulationConfig,
    active_pairs: set[frozenset[str]],
) -> dict[frozenset[str], np.ndarray]:
    available: dict[frozenset[str], np.ndarray] = {}
    for source in (
        _builtin_translation_periodic_vectors(config),
        _config_translation_periodic_vectors(config),
    ):
        for pair, vector in source.items():
            if pair in available:
                raise ValueError(
                    f"Duplicate OpenFOAM periodic pair metadata for {sorted(pair)}."
                )
            available[pair] = vector

    resolved: dict[frozenset[str], np.ndarray] = {}
    for pair in active_pairs:
        if pair not in available:
            raise ValueError(
                f"OpenFOAM backend does not support periodic pair {sorted(pair)} "
                f"on domain '{config.domain.type}'."
            )
        resolved[pair] = available[pair]
    return resolved


def _ensure_supported_expression(
    expr: FieldExpressionConfig,
    *,
    allow_quadrants: bool,
    allow_initial_only_types: bool = False,
    context: str,
) -> None:
    supported = {
        "none",
        "zero",
        "constant",
        "gaussian_bump",
        "radial_cosine",
        "affine",
        "step",
        "sine_waves",
    }
    if allow_quadrants:
        supported.add("quadrants")
    if allow_initial_only_types:
        supported.update({"gaussian_noise", "gaussian_blobs", "gaussian_wave_packet"})

    if expr.is_componentwise:
        for label, component in expr.components.items():
            _ensure_supported_expression(
                component,
                allow_quadrants=allow_quadrants,
                allow_initial_only_types=allow_initial_only_types,
                context=f"{context}.{label}",
            )
        return

    if expr.type not in supported:
        raise ValueError(
            f"{context} uses unsupported OpenFOAM expression type '{expr.type}'. "
            f"Supported types: {sorted(supported)}."
        )


def _scalar_field_cpp_expr(
    expr: FieldExpressionConfig,
    *,
    parameters: dict[str, float],
    gdim: int,
) -> str:
    if expr.is_componentwise:
        raise ValueError("Expected a scalar field expression.")
    if expr.type is None:
        raise ValueError("Scalar expression is missing a type.")

    params = expr.params
    expr_type = expr.type

    if expr_type in {"none", "zero"}:
        return "0.0"

    if expr_type == "constant":
        return _format_scalar(resolve_param_ref(params["value"], parameters))

    if expr_type == "gaussian_bump":
        amplitude = _format_scalar(resolve_param_ref(params["amplitude"], parameters))
        sigma = _format_scalar(resolve_param_ref(params["sigma"], parameters))
        center = [
            _format_scalar(resolve_param_ref(value, parameters))
            for value in params["center"][:gdim]
        ]
        radius_sq = " + ".join(
            f"Foam::sqr({coord} - {center_value})"
            for coord, center_value in zip(("x", "y", "z"), center, strict=False)
        )
        return f"({amplitude}*Foam::exp(-(({radius_sq})/(2.0*Foam::sqr({sigma})))))"

    if expr_type == "radial_cosine":
        base = _format_scalar(resolve_param_ref(params["base"], parameters))
        amplitude = _format_scalar(resolve_param_ref(params["amplitude"], parameters))
        frequency = _format_scalar(resolve_param_ref(params["frequency"], parameters))
        center = [
            _format_scalar(resolve_param_ref(value, parameters))
            for value in params["center"][:gdim]
        ]
        radius_sq = " + ".join(
            f"Foam::sqr({coord} - {center_value})"
            for coord, center_value in zip(("x", "y", "z"), center, strict=False)
        )
        return f"({base} + {amplitude}*Foam::cos({frequency}*Foam::sqrt({radius_sq})))"

    if expr_type == "affine":
        terms = [
            _format_scalar(resolve_param_ref(params.get("constant", 0.0), parameters))
        ]
        for label in component_labels_for_dim(gdim):
            if label not in params:
                continue
            terms.append(
                f"({_format_scalar(resolve_param_ref(params[label], parameters))}*{label})"
            )
        return "(" + " + ".join(terms) + ")"

    if expr_type == "step":
        axis = int(resolve_param_ref(params["axis"], parameters))
        coord = component_labels_for_dim(gdim)[axis]
        x_split = _format_scalar(resolve_param_ref(params["x_split"], parameters))
        value_left = _format_scalar(resolve_param_ref(params["value_left"], parameters))
        value_right = _format_scalar(
            resolve_param_ref(params["value_right"], parameters)
        )
        return f"(({coord} < {x_split}) ? {value_left} : {value_right})"

    if expr_type == "sine_waves":
        background = _format_scalar(resolve_param_ref(params["background"], parameters))
        modes: list[str] = []
        for mode in params["modes"]:
            amplitude = _format_scalar(resolve_param_ref(mode["amplitude"], parameters))
            phase = _format_scalar(resolve_param_ref(mode["phase"], parameters))
            angle = resolve_param_ref(mode.get("angle", 0.0), parameters)
            coord_exprs = ["x", "y", "z"]
            if gdim >= 2 and angle != 0.0:
                cos_angle = _format_scalar(float(np.cos(angle)))
                sin_angle = _format_scalar(float(np.sin(angle)))
                coord_exprs[0] = f"({cos_angle}*x - {sin_angle}*y)"
                coord_exprs[1] = f"({sin_angle}*x + {cos_angle}*y)"

            term = amplitude
            for axis, cycle_value in enumerate(mode["cycles"]):
                cycle = resolve_param_ref(cycle_value, parameters)
                if cycle == 0.0:
                    continue
                cycle_expr = _format_scalar(cycle)
                term = (
                    f"({term}*Foam::sin({_PI_LITERAL}*{cycle_expr}"
                    f"*{coord_exprs[axis]} + {phase}))"
                )
            modes.append(term)
        if not modes:
            return background
        return "(" + " + ".join([background, *modes]) + ")"

    if expr_type == "quadrants":
        split = [
            _format_scalar(resolve_param_ref(value, parameters))
            for value in params["split"][:gdim]
        ]
        cases: list[tuple[str, str]] = []
        for key, raw_value in sorted(params["region_values"].items()):
            value = _format_scalar(resolve_param_ref(raw_value, parameters))
            checks = []
            for axis, bit in enumerate(key):
                coord = component_labels_for_dim(gdim)[axis]
                threshold = split[axis]
                if bit == "0":
                    checks.append(f"({coord} < {threshold})")
                else:
                    checks.append(f"({coord} >= {threshold})")
            cases.append((" && ".join(checks), value))

        result = cases[-1][1]
        for condition, value in reversed(cases[:-1]):
            result = f"(({condition}) ? {value} : {result})"
        return result

    raise ValueError(
        f"OpenFOAM C++ code generation does not support type '{expr_type}'."
    )


def _vector_field_cpp_expr(
    expr: FieldExpressionConfig,
    *,
    parameters: dict[str, float],
    gdim: int,
) -> str:
    components = component_expressions(expr, gdim)
    values = [
        _scalar_field_cpp_expr(
            components[label],
            parameters=parameters,
            gdim=gdim,
        )
        for label in component_labels_for_dim(gdim)
    ]
    while len(values) < 3:
        values.append("0.0")
    return f"Foam::vector({values[0]}, {values[1]}, {values[2]})"


def _is_uniform_scalar_expression(
    expr: FieldExpressionConfig,
    *,
    parameters: dict[str, float],
) -> float | None:
    if expr.is_componentwise or expr.type is None:
        return None
    if expr.type in {"none", "zero"}:
        return 0.0
    if expr.type != "constant":
        return None
    return float(resolve_param_ref(expr.params["value"], parameters))


def _is_uniform_vector_expression(
    expr: FieldExpressionConfig,
    *,
    parameters: dict[str, float],
    gdim: int,
) -> np.ndarray | None:
    if not expr.is_componentwise and expr.type in {"none", "zero"}:
        return _pad_to_foam_vector(np.zeros(gdim, dtype=float))

    components = component_expressions(expr, gdim)
    values: list[float] = []
    for label in component_labels_for_dim(gdim):
        component = components[label]
        scalar_value = _is_uniform_scalar_expression(
            component,
            parameters=parameters,
        )
        if scalar_value is None:
            return None
        values.append(scalar_value)
    return _pad_to_foam_vector(np.asarray(values, dtype=float))
