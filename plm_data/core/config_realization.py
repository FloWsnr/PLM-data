"""Seeded realization of declarative simulation configs."""

from copy import deepcopy
from typing import Any

import numpy as np

from plm_data.core.config import (
    BoundaryConditionConfig,
    BoundaryFieldConfig,
    DomainConfig,
    FieldExpressionConfig,
    InputConfig,
    SimulationConfig,
)
from plm_data.core.sampling import (
    is_sampler_spec,
    rng_for_stream,
    sample_coordinate_list,
    sample_integer,
    sample_number,
)


def _realize_parameters(
    parameters: dict[str, Any], seed: int | None
) -> dict[str, float]:
    pending = {name: deepcopy(value) for name, value in parameters.items()}
    resolved: dict[str, float] = {}

    while pending:
        progressed = False
        for name in list(pending):
            value = pending[name]
            try:
                resolved[name] = sample_number(
                    value,
                    parameters=resolved,
                    rng=(
                        None
                        if seed is None
                        else rng_for_stream(seed, f"parameters.{name}")
                    ),
                    context=f"parameters.{name}",
                )
            except ValueError as exc:
                if "references unknown parameter" in str(exc):
                    continue
                raise
            del pending[name]
            progressed = True

        if progressed:
            continue

        raise ValueError(
            "Could not resolve sampled parameters or parameter references for "
            f"{sorted(pending)}."
        )

    return resolved


def _realize_numeric_tree(
    value: Any,
    *,
    parameters: dict[str, float],
    seed: int | None,
    stream_root: str,
    integer: bool = False,
) -> Any:
    if isinstance(value, list):
        return [
            _realize_numeric_tree(
                item,
                parameters=parameters,
                seed=seed,
                stream_root=f"{stream_root}[{index}]",
                integer=integer,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict) and not is_sampler_spec(value):
        return {
            key: _realize_numeric_tree(
                item,
                parameters=parameters,
                seed=seed,
                stream_root=f"{stream_root}.{key}",
                integer=integer,
            )
            for key, item in value.items()
        }
    if integer:
        return sample_integer(
            value,
            parameters=parameters,
            rng=None if seed is None else rng_for_stream(seed, stream_root),
            context=stream_root,
        )
    return sample_number(
        value,
        parameters=parameters,
        rng=None if seed is None else rng_for_stream(seed, stream_root),
        context=stream_root,
    )


def _realize_scalar_expression(
    expr: FieldExpressionConfig,
    *,
    gdim: int,
    parameters: dict[str, float],
    seed: int | None,
    stream_root: str,
) -> FieldExpressionConfig:
    if expr.is_componentwise:
        raise ValueError("Expected a scalar expression, got a component-wise config.")
    if expr.type is None:
        raise ValueError("Scalar field expression must define a 'type'.")

    expr_type = expr.type
    params = expr.params
    rng = None if seed is None else rng_for_stream(seed, stream_root)

    if expr_type in {"none", "zero", "custom"}:
        return FieldExpressionConfig(type=expr_type, params=deepcopy(params))

    if expr_type == "constant":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "value": sample_number(
                    params["value"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.value",
                )
            },
        )

    if expr_type == "gaussian_bump":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "amplitude": sample_number(
                    params["amplitude"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.amplitude",
                ),
                "sigma": sample_number(
                    params["sigma"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.sigma",
                ),
                "center": sample_coordinate_list(
                    params["center"],
                    gdim=gdim,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.center",
                ),
            },
        )

    if expr_type == "radial_cosine":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "base": sample_number(
                    params["base"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.base",
                ),
                "amplitude": sample_number(
                    params["amplitude"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.amplitude",
                ),
                "frequency": sample_number(
                    params["frequency"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.frequency",
                ),
                "center": sample_coordinate_list(
                    params["center"],
                    gdim=gdim,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.center",
                ),
            },
        )

    if expr_type == "affine":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                key: sample_number(
                    value,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.{key}",
                )
                for key, value in params.items()
            },
        )

    if expr_type == "step":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "value_left": sample_number(
                    params["value_left"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.value_left",
                ),
                "value_right": sample_number(
                    params["value_right"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.value_right",
                ),
                "x_split": sample_number(
                    params["x_split"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.x_split",
                ),
                "axis": sample_integer(
                    params["axis"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.axis",
                ),
            },
        )

    if expr_type == "gaussian_noise":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "mean": sample_number(
                    params["mean"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.mean",
                ),
                "std": sample_number(
                    params["std"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.std",
                ),
            },
        )

    if expr_type == "gaussian_blobs":
        if "blobs" in params:
            return FieldExpressionConfig(type=expr_type, params=deepcopy(params))

        realized_blobs: list[dict[str, Any]] = []
        for generator_index, generator in enumerate(params["generators"]):
            count = sample_integer(
                generator["count"],
                parameters=parameters,
                rng=rng,
                context=f"{stream_root}.params.generators[{generator_index}].count",
            )
            if count <= 0:
                raise ValueError(
                    f"{stream_root}.params.generators[{generator_index}].count must "
                    f"be positive. Got {count}."
                )
            for blob_index in range(count):
                aspect_ratio = sample_number(
                    generator["aspect_ratio"],
                    parameters=parameters,
                    rng=rng,
                    context=(
                        f"{stream_root}.params.generators[{generator_index}]"
                        f".aspect_ratio[{blob_index}]"
                    ),
                )
                if aspect_ratio == 1.0:
                    direction = np.zeros(gdim)
                    direction[0] = 1.0
                else:
                    if rng is None:
                        raise ValueError(
                            f"{stream_root} requires an explicit seed from the "
                            "config or '--seed'."
                        )
                    direction = rng.standard_normal(gdim)
                    norm = float(np.linalg.norm(direction))
                    if norm < 1.0e-12:
                        direction = np.zeros(gdim)
                        direction[0] = 1.0
                    else:
                        direction = direction / norm

                realized_blobs.append(
                    {
                        "amplitude": sample_number(
                            generator["amplitude"],
                            parameters=parameters,
                            rng=rng,
                            context=(
                                f"{stream_root}.params.generators[{generator_index}]"
                                f".amplitude[{blob_index}]"
                            ),
                        ),
                        "sigma": sample_number(
                            generator["sigma"],
                            parameters=parameters,
                            rng=rng,
                            context=(
                                f"{stream_root}.params.generators[{generator_index}]"
                                f".sigma[{blob_index}]"
                            ),
                        ),
                        "center": sample_coordinate_list(
                            generator["center"],
                            gdim=gdim,
                            parameters=parameters,
                            rng=rng,
                            context=(
                                f"{stream_root}.params.generators[{generator_index}]"
                                f".center[{blob_index}]"
                            ),
                        ),
                        "aspect_ratio": aspect_ratio,
                        "direction": direction.tolist(),
                    }
                )

        return FieldExpressionConfig(
            type=expr_type,
            params={
                "background": sample_number(
                    params["background"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.background",
                ),
                "blobs": realized_blobs,
            },
        )

    if expr_type == "gaussian_wave_packet":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "amplitude": sample_number(
                    params["amplitude"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.amplitude",
                ),
                "sigma": sample_number(
                    params["sigma"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.sigma",
                ),
                "center": sample_coordinate_list(
                    params["center"],
                    gdim=gdim,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.center",
                ),
                "wavevector": sample_coordinate_list(
                    params["wavevector"],
                    gdim=gdim,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.wavevector",
                ),
                "phase": sample_number(
                    params["phase"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.phase",
                ),
            },
        )

    if expr_type == "sine_waves":
        realized_modes = []
        for mode_index, mode in enumerate(params["modes"]):
            realized_modes.append(
                {
                    "amplitude": sample_number(
                        mode["amplitude"],
                        parameters=parameters,
                        rng=rng,
                        context=f"{stream_root}.params.modes[{mode_index}].amplitude",
                    ),
                    "cycles": [
                        sample_number(
                            cycle,
                            parameters=parameters,
                            rng=rng,
                            context=(
                                f"{stream_root}.params.modes[{mode_index}]"
                                f".cycles[{axis}]"
                            ),
                        )
                        for axis, cycle in enumerate(mode["cycles"])
                    ],
                    "phase": sample_number(
                        mode["phase"],
                        parameters=parameters,
                        rng=rng,
                        context=f"{stream_root}.params.modes[{mode_index}].phase",
                    ),
                    "angle": sample_number(
                        mode.get("angle", 0.0),
                        parameters=parameters,
                        rng=rng,
                        context=f"{stream_root}.params.modes[{mode_index}].angle",
                    ),
                }
            )
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "background": sample_number(
                    params["background"],
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.background",
                ),
                "modes": realized_modes,
            },
        )

    if expr_type == "quadrants":
        return FieldExpressionConfig(
            type=expr_type,
            params={
                "split": sample_coordinate_list(
                    params["split"],
                    gdim=gdim,
                    parameters=parameters,
                    rng=rng,
                    context=f"{stream_root}.params.split",
                ),
                "region_values": {
                    key: sample_number(
                        value,
                        parameters=parameters,
                        rng=rng,
                        context=f"{stream_root}.params.region_values.{key}",
                    )
                    for key, value in params["region_values"].items()
                },
            },
        )

    raise ValueError(
        f"{stream_root} uses unsupported field expression type '{expr_type}'."
    )


def _realize_field_expression(
    expr: FieldExpressionConfig,
    *,
    gdim: int,
    parameters: dict[str, float],
    seed: int | None,
    stream_root: str,
) -> FieldExpressionConfig:
    if expr.is_componentwise:
        return FieldExpressionConfig(
            components={
                label: _realize_scalar_expression(
                    component,
                    gdim=gdim,
                    parameters=parameters,
                    seed=seed,
                    stream_root=f"{stream_root}.components.{label}",
                )
                for label, component in expr.components.items()
            }
        )
    return _realize_scalar_expression(
        expr,
        gdim=gdim,
        parameters=parameters,
        seed=seed,
        stream_root=stream_root,
    )


def _realize_domain_params(
    domain: DomainConfig,
    *,
    parameters: dict[str, float],
    seed: int | None,
) -> dict[str, Any]:
    params = domain.params
    realized: dict[str, Any] = {}

    if domain.type == "interval":
        size = params["size"]
        if isinstance(size, list):
            realized["size"] = _realize_numeric_tree(
                size,
                parameters=parameters,
                seed=seed,
                stream_root="domain.params.size",
            )
        else:
            realized["size"] = _realize_numeric_tree(
                size,
                parameters=parameters,
                seed=seed,
                stream_root="domain.params.size",
            )
        mesh_resolution = params["mesh_resolution"]
        if isinstance(mesh_resolution, list):
            realized["mesh_resolution"] = _realize_numeric_tree(
                mesh_resolution,
                parameters=parameters,
                seed=seed,
                stream_root="domain.params.mesh_resolution",
                integer=True,
            )
        else:
            realized["mesh_resolution"] = _realize_numeric_tree(
                mesh_resolution,
                parameters=parameters,
                seed=seed,
                stream_root="domain.params.mesh_resolution",
                integer=True,
            )
    elif domain.type in {"rectangle", "box"}:
        realized["size"] = _realize_numeric_tree(
            params["size"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.size",
        )
        realized["mesh_resolution"] = _realize_numeric_tree(
            params["mesh_resolution"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.mesh_resolution",
            integer=True,
        )
    elif domain.type == "disk":
        realized["center"] = _realize_numeric_tree(
            params["center"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.center",
        )
        realized["radius"] = _realize_numeric_tree(
            params["radius"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.radius",
        )
        realized["mesh_size"] = _realize_numeric_tree(
            params["mesh_size"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.mesh_size",
        )
    elif domain.type == "dumbbell":
        for key in (
            "left_center",
            "right_center",
            "lobe_radius",
            "neck_width",
            "mesh_size",
        ):
            realized[key] = _realize_numeric_tree(
                params[key],
                parameters=parameters,
                seed=seed,
                stream_root=f"domain.params.{key}",
            )
    elif domain.type == "parallelogram":
        for key in ("origin", "axis_x", "axis_y"):
            realized[key] = _realize_numeric_tree(
                params[key],
                parameters=parameters,
                seed=seed,
                stream_root=f"domain.params.{key}",
            )
        realized["mesh_resolution"] = _realize_numeric_tree(
            params["mesh_resolution"],
            parameters=parameters,
            seed=seed,
            stream_root="domain.params.mesh_resolution",
            integer=True,
        )
    elif domain.type == "channel_obstacle":
        for key in (
            "length",
            "height",
            "obstacle_center",
            "obstacle_radius",
            "mesh_size",
        ):
            realized[key] = _realize_numeric_tree(
                params[key],
                parameters=parameters,
                seed=seed,
                stream_root=f"domain.params.{key}",
            )
    elif domain.type == "annulus":
        for key in ("inner_radius", "outer_radius", "mesh_size"):
            realized[key] = _realize_numeric_tree(
                params[key],
                parameters=parameters,
                seed=seed,
                stream_root=f"domain.params.{key}",
            )

    for key, value in params.items():
        if key in realized:
            continue
        realized[key] = _realize_numeric_tree(
            value,
            parameters=parameters,
            seed=seed,
            stream_root=f"domain.params.{key}",
        )

    return realized


def realize_simulation_config(
    config: SimulationConfig,
    *,
    realize_initial_conditions: bool = True,
) -> SimulationConfig:
    seed = config.seed
    parameters = _realize_parameters(config.parameters, seed)
    domain = DomainConfig(
        type=config.domain.type,
        params=_realize_domain_params(config.domain, parameters=parameters, seed=seed),
        periodic_maps=deepcopy(config.domain.periodic_maps),
        allow_sampling=config.domain.allow_sampling,
    )
    gdim = domain.dimension

    inputs = {
        name: InputConfig(
            source=(
                None
                if input_config.source is None
                else _realize_field_expression(
                    input_config.source,
                    gdim=gdim,
                    parameters=parameters,
                    seed=seed,
                    stream_root=f"inputs.{name}.source",
                )
            ),
            initial_condition=(
                None
                if input_config.initial_condition is None
                else (
                    _realize_field_expression(
                        input_config.initial_condition,
                        gdim=gdim,
                        parameters=parameters,
                        seed=seed,
                        stream_root=f"inputs.{name}.initial_condition",
                    )
                    if realize_initial_conditions
                    else deepcopy(input_config.initial_condition)
                )
            ),
        )
        for name, input_config in config.inputs.items()
    }

    coefficients = {
        name: _realize_field_expression(
            coefficient,
            gdim=gdim,
            parameters=parameters,
            seed=seed,
            stream_root=f"coefficients.{name}",
        )
        for name, coefficient in config.coefficients.items()
    }

    boundary_conditions = {}
    for field_name, boundary_field in config.boundary_conditions.items():
        sides = {}
        for side_name, entries in boundary_field.sides.items():
            realized_entries = []
            for index, entry in enumerate(entries):
                stream_root = f"boundary_conditions.{field_name}.{side_name}[{index}]"
                realized_entries.append(
                    BoundaryConditionConfig(
                        type=entry.type,
                        value=(
                            None
                            if entry.value is None
                            else _realize_field_expression(
                                entry.value,
                                gdim=gdim,
                                parameters=parameters,
                                seed=seed,
                                stream_root=f"{stream_root}.value",
                            )
                        ),
                        pair_with=entry.pair_with,
                        operator_parameters={
                            key: sample_number(
                                value,
                                parameters=parameters,
                                rng=(
                                    None
                                    if seed is None
                                    else rng_for_stream(
                                        seed,
                                        f"{stream_root}.operator_parameters.{key}",
                                    )
                                ),
                                context=f"{stream_root}.operator_parameters.{key}",
                            )
                            for key, value in entry.operator_parameters.items()
                        },
                    )
                )
            sides[side_name] = realized_entries
        boundary_conditions[field_name] = BoundaryFieldConfig(sides=sides)

    return SimulationConfig(
        preset=config.preset,
        parameters=parameters,
        domain=domain,
        inputs=inputs,
        boundary_conditions=boundary_conditions,
        output=deepcopy(config.output),
        solver=deepcopy(config.solver),
        time=deepcopy(config.time),
        seed=config.seed,
        coefficients=coefficients,
        stochastic=deepcopy(config.stochastic),
    )
