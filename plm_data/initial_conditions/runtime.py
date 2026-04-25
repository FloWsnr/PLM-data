"""Initial condition helpers for scalar and vector fields."""

from typing import Any, Callable

import numpy as np
from dolfinx import fem

from plm_data.core.runtime_config import FieldExpressionConfig
from plm_data.sampling.values import (
    rng_for_stream,
    sample_coordinate_list,
    sample_integer,
    sample_number,
)
from plm_data.fields import (
    build_interpolator,
    build_vector_interpolator,
    component_expressions,
    component_labels_for_dim,
)


def _require_param(params: dict[str, Any], key: str, field_type: str) -> Any:
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for field type '{field_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def _resolved_scalar_ic(
    ic_config: FieldExpressionConfig,
    *,
    gdim: int,
    parameters: dict[str, float],
    rng: np.random.Generator | None,
) -> tuple[str, dict[str, Any]]:
    if ic_config.is_componentwise:
        raise ValueError("Expected a scalar initial-condition config.")
    if ic_config.type is None:
        raise ValueError("Scalar initial-condition config must define a 'type'.")

    ic_type = ic_config.type
    params = ic_config.params

    if ic_type in {"none", "zero", "custom"}:
        return ic_type, dict(params)

    if ic_type == "constant":
        return ic_type, {
            "value": sample_number(
                _require_param(params, "value", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.value",
            )
        }

    if ic_type == "gaussian_bump":
        return ic_type, {
            "amplitude": sample_number(
                _require_param(params, "amplitude", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.amplitude",
            ),
            "sigma": sample_number(
                _require_param(params, "sigma", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.sigma",
            ),
            "center": sample_coordinate_list(
                _require_param(params, "center", ic_type),
                gdim=gdim,
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.center",
            ),
        }

    if ic_type == "radial_cosine":
        return ic_type, {
            "base": sample_number(
                _require_param(params, "base", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.base",
            ),
            "amplitude": sample_number(
                _require_param(params, "amplitude", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.amplitude",
            ),
            "frequency": sample_number(
                _require_param(params, "frequency", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.frequency",
            ),
            "center": sample_coordinate_list(
                _require_param(params, "center", ic_type),
                gdim=gdim,
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.center",
            ),
        }

    if ic_type == "affine":
        resolved_params = {}
        for key in ("constant", *component_labels_for_dim(gdim)):
            if key in params:
                resolved_params[key] = sample_number(
                    params[key],
                    parameters=parameters,
                    rng=rng,
                    context=f"{ic_type}.{key}",
                )
        return ic_type, resolved_params

    if ic_type == "step":
        return ic_type, {
            "value_left": sample_number(
                _require_param(params, "value_left", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.value_left",
            ),
            "value_right": sample_number(
                _require_param(params, "value_right", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.value_right",
            ),
            "x_split": sample_number(
                _require_param(params, "x_split", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.x_split",
            ),
            "axis": sample_integer(
                _require_param(params, "axis", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.axis",
            ),
        }

    if ic_type == "gaussian_noise":
        return ic_type, {
            "mean": sample_number(
                _require_param(params, "mean", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.mean",
            ),
            "std": sample_number(
                _require_param(params, "std", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.std",
            ),
        }

    if ic_type == "gaussian_blobs":
        if "blobs" in params:
            return ic_type, {
                "background": sample_number(
                    _require_param(params, "background", ic_type),
                    parameters=parameters,
                    rng=rng,
                    context=f"{ic_type}.background",
                ),
                "blobs": [
                    {
                        "amplitude": float(blob["amplitude"]),
                        "sigma": float(blob["sigma"]),
                        "center": [float(value) for value in blob["center"]],
                        "aspect_ratio": float(blob["aspect_ratio"]),
                        "direction": [float(value) for value in blob["direction"]],
                    }
                    for blob in _require_param(params, "blobs", ic_type)
                ],
            }
        raw_generators = _require_param(params, "generators", ic_type)
        if not isinstance(raw_generators, list) or not raw_generators:
            raise ValueError("gaussian_blobs requires a non-empty 'generators' list.")

        resolved_blobs = []
        for generator in raw_generators:
            if not isinstance(generator, dict):
                raise ValueError(
                    "gaussian_blobs generators must be mappings with count, "
                    "amplitude, sigma, and center."
                )
            count = sample_integer(
                _require_param(generator, "count", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.count",
            )
            if count <= 0:
                raise ValueError(
                    f"gaussian_blobs generator count must be positive. Got {count}."
                )
            for _ in range(count):
                aspect_ratio = sample_number(
                    _require_param(generator, "aspect_ratio", ic_type),
                    parameters=parameters,
                    rng=rng,
                    context=f"{ic_type}.aspect_ratio",
                )
                # Random elongation direction (unit vector); rotation is
                # always random and not configurable.  When aspect_ratio
                # is exactly 1 the direction has no effect, so a fixed
                # axis-aligned vector is used (avoids requiring an rng
                # for non-sampled circular blobs).
                if aspect_ratio == 1.0:
                    raw_dir = np.zeros(gdim)
                    raw_dir[0] = 1.0
                else:
                    if rng is None:
                        raise ValueError(
                            "Elliptical gaussian_blobs require an explicit seed "
                            "from the config or '--seed'."
                        )
                    raw_dir = rng.standard_normal(gdim)
                    norm = float(np.linalg.norm(raw_dir))
                    if norm < 1e-12:
                        raw_dir = np.zeros(gdim)
                        raw_dir[0] = 1.0
                    else:
                        raw_dir = raw_dir / norm
                resolved_blobs.append(
                    {
                        "amplitude": sample_number(
                            _require_param(generator, "amplitude", ic_type),
                            parameters=parameters,
                            rng=rng,
                            context=f"{ic_type}.amplitude",
                        ),
                        "sigma": sample_number(
                            _require_param(generator, "sigma", ic_type),
                            parameters=parameters,
                            rng=rng,
                            context=f"{ic_type}.sigma",
                        ),
                        "center": sample_coordinate_list(
                            _require_param(generator, "center", ic_type),
                            gdim=gdim,
                            parameters=parameters,
                            rng=rng,
                            context=f"{ic_type}.center",
                        ),
                        "aspect_ratio": aspect_ratio,
                        "direction": raw_dir.tolist(),
                    }
                )

        return ic_type, {
            "background": sample_number(
                _require_param(params, "background", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.background",
            ),
            "blobs": resolved_blobs,
        }

    if ic_type == "gaussian_wave_packet":
        return ic_type, {
            "amplitude": sample_number(
                _require_param(params, "amplitude", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.amplitude",
            ),
            "sigma": sample_number(
                _require_param(params, "sigma", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.sigma",
            ),
            "center": sample_coordinate_list(
                _require_param(params, "center", ic_type),
                gdim=gdim,
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.center",
            ),
            "wavevector": sample_coordinate_list(
                _require_param(params, "wavevector", ic_type),
                gdim=gdim,
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.wavevector",
            ),
            "phase": sample_number(
                _require_param(params, "phase", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.phase",
            ),
        }

    if ic_type == "sine_waves":
        raw_modes = _require_param(params, "modes", ic_type)
        if not isinstance(raw_modes, list) or not raw_modes:
            raise ValueError("sine_waves requires a non-empty 'modes' list.")

        resolved_modes = []
        for mode in raw_modes:
            if not isinstance(mode, dict):
                raise ValueError(
                    "sine_waves modes must be mappings with amplitude, cycles, "
                    "and phase."
                )
            cycles_raw = _require_param(mode, "cycles", ic_type)
            if not isinstance(cycles_raw, list) or len(cycles_raw) != gdim:
                raise ValueError(
                    f"sine_waves cycles must have {gdim} entries in {gdim}D. "
                    f"Got {cycles_raw!r}."
                )
            resolved_modes.append(
                {
                    "amplitude": sample_number(
                        _require_param(mode, "amplitude", ic_type),
                        parameters=parameters,
                        rng=rng,
                        context=f"{ic_type}.amplitude",
                    ),
                    "cycles": [
                        sample_number(
                            cycle_value,
                            parameters=parameters,
                            rng=rng,
                            context=f"{ic_type}.cycles[{axis}]",
                        )
                        for axis, cycle_value in enumerate(cycles_raw)
                    ],
                    "phase": sample_number(
                        _require_param(mode, "phase", ic_type),
                        parameters=parameters,
                        rng=rng,
                        context=f"{ic_type}.phase",
                    ),
                    "angle": sample_number(
                        mode.get("angle", 0.0),
                        parameters=parameters,
                        rng=rng,
                        context=f"{ic_type}.angle",
                    ),
                }
            )

        return ic_type, {
            "background": sample_number(
                _require_param(params, "background", ic_type),
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.background",
            ),
            "modes": resolved_modes,
        }

    if ic_type == "quadrants":
        raw_region_values = _require_param(params, "region_values", ic_type)
        if not isinstance(raw_region_values, dict):
            raise ValueError("quadrants region_values must be a mapping.")

        expected_keys = {format(index, f"0{gdim}b") for index in range(2**gdim)}
        if set(raw_region_values) != expected_keys:
            raise ValueError(
                f"quadrants region_values must contain exactly "
                f"{sorted(expected_keys)}. Got {sorted(raw_region_values)}."
            )

        return ic_type, {
            "split": sample_coordinate_list(
                _require_param(params, "split", ic_type),
                gdim=gdim,
                parameters=parameters,
                rng=rng,
                context=f"{ic_type}.split",
            ),
            "region_values": {
                key: sample_number(
                    value,
                    parameters=parameters,
                    rng=rng,
                    context=f"{ic_type}.region_values.{key}",
                )
                for key, value in raw_region_values.items()
            },
        }

    raise ValueError(f"Unknown initial-condition type: '{ic_type}'")


def _build_gaussian_blobs_interpolator(
    *,
    background: float,
    blobs: list[dict[str, Any]],
) -> Callable[[np.ndarray], np.ndarray]:
    blob_parameters = [
        (
            float(blob["amplitude"]),
            float(blob["sigma"]),
            np.asarray(blob["center"], dtype=float),
            float(blob["aspect_ratio"]),
            np.asarray(blob["direction"], dtype=float),
        )
        for blob in blobs
    ]

    def _interpolator(x: np.ndarray) -> np.ndarray:
        values = np.full(x.shape[1], background, dtype=float)
        for amplitude, sigma, center, aspect_ratio, direction in blob_parameters:
            ndim = center.shape[0]
            dx = x[:ndim, :] - center[:, None]
            total_sq = np.sum(dx**2, axis=0)
            parallel_sq = np.dot(direction, dx) ** 2
            perp_sq = total_sq - parallel_sq
            r_eff_sq = parallel_sq + aspect_ratio**2 * perp_sq
            values = values + amplitude * np.exp(-r_eff_sq / (2.0 * sigma**2))
        return values

    return _interpolator


def _build_gaussian_wave_packet_interpolator(
    *,
    amplitude: float,
    sigma: float,
    center: list[float],
    wavevector: list[float],
    phase: float,
) -> Callable[[np.ndarray], np.ndarray]:
    center_array = np.asarray(center, dtype=float)
    wavevector_array = np.asarray(wavevector, dtype=float)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        shifted = x[: center_array.shape[0], :] - center_array[:, None]
        radius_squared = np.sum(shifted**2, axis=0)
        envelope = amplitude * np.exp(-radius_squared / (2.0 * sigma**2))
        carrier_phase = wavevector_array @ shifted + phase
        return envelope * np.cos(carrier_phase)

    return _interpolator


def _build_quadrants_interpolator(
    *,
    split: list[float],
    region_values: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    split_array = np.asarray(split, dtype=float)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        values = np.zeros(x.shape[1], dtype=float)
        for key, region_value in region_values.items():
            mask = np.ones(x.shape[1], dtype=bool)
            for axis, bit in enumerate(key):
                if bit == "0":
                    mask &= x[axis] < split_array[axis]
                else:
                    mask &= x[axis] >= split_array[axis]
            values[mask] = region_value
        return values

    return _interpolator


def _build_resolved_scalar_interpolator(
    *,
    msh,
    ic_type: str,
    resolved_params: dict[str, Any],
    rng: np.random.Generator | None,
    parameters: dict[str, float],
) -> Callable[[np.ndarray], np.ndarray] | None:
    if ic_type == "custom":
        return None

    if ic_type == "gaussian_noise":
        if rng is None:
            raise ValueError(
                "Initial-condition gaussian_noise requires an explicit seed "
                "from the config or '--seed'."
            )
        mean = float(resolved_params["mean"])
        std = float(resolved_params["std"])

        def _noise(
            x: np.ndarray,
            local_rng: np.random.Generator = rng,
            mean_value: float = mean,
            std_value: float = std,
        ) -> np.ndarray:
            return local_rng.normal(mean_value, std_value, size=x.shape[1])

        return _noise

    if ic_type == "gaussian_blobs":
        return _build_gaussian_blobs_interpolator(
            background=float(resolved_params["background"]),
            blobs=list(resolved_params["blobs"]),
        )

    if ic_type == "gaussian_wave_packet":
        return _build_gaussian_wave_packet_interpolator(
            amplitude=float(resolved_params["amplitude"]),
            sigma=float(resolved_params["sigma"]),
            center=list(resolved_params["center"]),
            wavevector=list(resolved_params["wavevector"]),
            phase=float(resolved_params["phase"]),
        )

    if ic_type == "quadrants":
        return _build_quadrants_interpolator(
            split=list(resolved_params["split"]),
            region_values=dict(resolved_params["region_values"]),
        )

    return build_interpolator(
        {"type": ic_type, "params": resolved_params},
        parameters,
    )


def build_scalar_ic_interpolator(
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    *,
    gdim: int,
    seed: int | None = None,
    stream_id: str | None = None,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a scalar initial-condition interpolator without a FEM function."""
    if ic_config.is_componentwise:
        raise ValueError("Expected a scalar initial-condition config.")

    rng = rng_for_stream(seed, stream_id) if seed is not None else None
    ic_type, resolved_params = _resolved_scalar_ic(
        ic_config,
        gdim=gdim,
        parameters=parameters,
        rng=rng,
    )
    return _build_resolved_scalar_interpolator(
        msh=None,
        ic_type=ic_type,
        resolved_params=resolved_params,
        rng=rng,
        parameters=parameters,
    )


def build_vector_ic_interpolator(
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    *,
    gdim: int,
    seed: int | None = None,
    stream_id: str | None = None,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Build a vector initial-condition interpolator without a FEM function."""
    if ic_config.type == "custom" and not ic_config.is_componentwise:
        return None

    stream_root = stream_id or "vector"
    if not ic_config.is_componentwise:
        return build_vector_interpolator(ic_config, gdim, parameters)

    components = component_expressions(ic_config, gdim)
    scalar_interpolators = []
    for label in component_labels_for_dim(gdim):
        component = components[label]
        if component.type == "custom":
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        component_rng = (
            rng_for_stream(seed, f"{stream_root}.{label}") if seed is not None else None
        )
        ic_type, resolved_params = _resolved_scalar_ic(
            component,
            gdim=gdim,
            parameters=parameters,
            rng=component_rng,
        )
        interpolator = _build_resolved_scalar_interpolator(
            msh=None,
            ic_type=ic_type,
            resolved_params=resolved_params,
            rng=component_rng,
            parameters=parameters,
        )
        if interpolator is None:
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        scalar_interpolators.append(interpolator)

    def _vector_interpolator(x: np.ndarray) -> np.ndarray:
        values = np.zeros((gdim, x.shape[1]))
        for i, interpolator in enumerate(scalar_interpolators):
            values[i, :] = interpolator(x)
        return values

    return _vector_interpolator


def apply_ic(
    func: fem.Function,
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    seed: int | None = None,
    *,
    stream_id: str | None = None,
) -> None:
    """Apply a scalar initial condition to a DOLFINx function in-place."""
    if ic_config.is_componentwise:
        raise ValueError("apply_ic expects a scalar initial-condition config")

    gdim = func.function_space.mesh.geometry.dim
    rng = (
        rng_for_stream(seed, stream_id or func.name or None)
        if seed is not None
        else None
    )
    ic_type, resolved_params = _resolved_scalar_ic(
        ic_config,
        gdim=gdim,
        parameters=parameters,
        rng=rng,
    )

    interpolator = _build_resolved_scalar_interpolator(
        msh=func.function_space.mesh,
        ic_type=ic_type,
        resolved_params=resolved_params,
        rng=rng,
        parameters=parameters,
    )
    if interpolator is not None:
        func.interpolate(interpolator)


def apply_vector_ic(
    func: fem.Function,
    ic_config: FieldExpressionConfig,
    parameters: dict[str, float],
    seed: int | None = None,
    *,
    stream_id: str | None = None,
) -> None:
    """Apply a vector initial condition to a DOLFINx vector function."""
    if ic_config.type == "custom" and not ic_config.is_componentwise:
        return

    gdim = func.function_space.mesh.geometry.dim
    stream_root = stream_id or func.name or "vector"
    if not ic_config.is_componentwise:
        interpolator = build_vector_interpolator(ic_config, gdim, parameters)
        if interpolator is not None:
            func.interpolate(interpolator)
        return

    components = component_expressions(ic_config, gdim)
    scalar_interpolators = []
    for label in component_labels_for_dim(gdim):
        component = components[label]
        if component.type == "custom":
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        component_rng = (
            rng_for_stream(seed, f"{stream_root}.{label}") if seed is not None else None
        )
        ic_type, resolved_params = _resolved_scalar_ic(
            component,
            gdim=gdim,
            parameters=parameters,
            rng=component_rng,
        )
        interpolator = _build_resolved_scalar_interpolator(
            msh=func.function_space.mesh,
            ic_type=ic_type,
            resolved_params=resolved_params,
            rng=component_rng,
            parameters=parameters,
        )
        if interpolator is None:
            raise ValueError(
                f"Component '{label}' cannot use 'custom' inside a vector expression"
            )
        scalar_interpolators.append(interpolator)

    def _vector_interpolator(x: np.ndarray) -> np.ndarray:
        values = np.zeros((gdim, x.shape[1]))
        for i, interp in enumerate(scalar_interpolators):
            values[i, :] = interp(x)
        return values

    func.interpolate(_vector_interpolator)
