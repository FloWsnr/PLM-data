"""Domain configuration validation and dimension inference."""

import math
from typing import Any, Protocol

import numpy as np

from plm_data.core.sampling import is_param_ref, is_sampler_spec
from plm_data.domains.airfoil import symmetric_naca_airfoil_outline


class DomainConfigLike(Protocol):
    """Structural domain config interface used by domain builders."""

    type: str
    params: dict[str, Any]
    periodic_maps: dict[str, Any]


def _require(raw: dict[str, Any], key: str, context: str = "config") -> Any:
    if key not in raw:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return raw[key]


def _as_mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping. Got: {raw!r}")
    return raw


def _is_param_ref(value: Any) -> bool:
    return is_param_ref(value)


def _validate_numeric_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, (int, float)) or _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be a number or 'param:<name>' reference. Got {value!r}."
    )


def _validate_integer_literal_or_param_ref(value: Any, context: str) -> None:
    if isinstance(value, int):
        return
    if isinstance(value, float) and value.is_integer():
        return
    if _is_param_ref(value):
        return
    raise ValueError(
        f"{context} must be an integer or 'param:<name>' reference. Got {value!r}."
    )


def _validate_sampleable_numeric(value: Any, context: str) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_numeric_literal_or_param_ref(value, context)
        return

    sample_type = _require(value, "sample", context)
    if sample_type == "uniform":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} uniform sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    if sample_type == "normal":
        if set(value) != {"sample", "mean", "std"}:
            raise ValueError(
                f"{context} normal sampler must contain exactly ['mean', 'sample', "
                f"'std']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["mean"], f"{context}.mean")
        _validate_numeric_literal_or_param_ref(value["std"], f"{context}.std")
        return

    if sample_type == "randint":
        if set(value) != {"sample", "min", "max"}:
            raise ValueError(
                f"{context} randint sampler must contain exactly ['max', 'min', "
                f"'sample']. Got {sorted(value)}."
            )
        _validate_numeric_literal_or_param_ref(value["min"], f"{context}.min")
        _validate_numeric_literal_or_param_ref(value["max"], f"{context}.max")
        return

    raise ValueError(f"{context} uses unknown sampler '{sample_type}'.")


def _validate_sampleable_integer(value: Any, context: str) -> None:
    if not isinstance(value, dict) or "sample" not in value:
        _validate_integer_literal_or_param_ref(value, context)
        return
    _validate_sampleable_numeric(value, context)


def infer_domain_dimension(domain_type: str, params: dict[str, Any]) -> int:
    """Infer the spatial dimension from the configured domain."""
    try:
        from plm_data.domains import get_domain_spec

        return get_domain_spec(domain_type).dimension
    except ValueError:
        pass

    builtin_dims = {
        "interval": 1,
        "rectangle": 2,
        "box": 3,
        "annulus": 2,
        "disk": 2,
        "dumbbell": 2,
        "l_shape": 2,
        "multi_hole_plate": 2,
        "parallelogram": 2,
        "channel_obstacle": 2,
        "y_bifurcation": 2,
        "venturi_channel": 2,
        "porous_channel": 2,
        "serpentine_channel": 2,
        "airfoil_channel": 2,
        "side_cavity_channel": 2,
    }
    if domain_type in builtin_dims:
        return builtin_dims[domain_type]

    size = params.get("size")
    if isinstance(size, list):
        return len(size)

    raise ValueError(
        f"Cannot infer spatial dimension for domain type '{domain_type}'. "
        "Provide a supported built-in domain."
    )


def validate_domain_params(
    domain_type: str,
    params: dict[str, Any],
    *,
    allow_sampling: bool = False,
) -> None:
    """Validate built-in domain parameters before mesh creation."""

    def _require_keys(*required: str) -> None:
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(
                f"{domain_type.capitalize()} domain requires parameters "
                f"{sorted(required)}. Missing {sorted(missing)}."
            )

    def _validate_domain_sampler(value: Any, context: str, *, integer: bool) -> None:
        if not is_sampler_spec(value):
            return
        sample_type = _require(value, "sample", context)
        if sample_type not in {"uniform", "randint"}:
            raise ValueError(
                f"{context} domain sampling supports only 'uniform' and 'randint'."
            )
        if integer and sample_type != "randint":
            raise ValueError(f"{context} must use 'randint' for integer sampling.")

    def _float_value(raw: Any, context: str, *, positive: bool = False) -> float | None:
        if is_sampler_spec(raw):
            if not allow_sampling:
                raise ValueError(
                    f"{context} uses sampled values, but domain sampling is disabled. "
                    "Set 'domain.allow_sampling: true' to enable it."
                )
            _validate_domain_sampler(raw, context, integer=False)
            _validate_sampleable_numeric(raw, context)
            return None
        if _is_param_ref(raw):
            return None
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{context} must be finite. Got {value}.")
        if positive and value <= 0.0:
            raise ValueError(f"{context} must be > 0. Got {value}.")
        return value

    def _int_param(name: str, *, minimum: int) -> int | None:
        raw = params[name]
        context = f"{domain_type.capitalize()} domain parameter '{name}'"
        if is_sampler_spec(raw):
            if not allow_sampling:
                raise ValueError(
                    f"{context} uses sampled values, but domain sampling is disabled. "
                    "Set 'domain.allow_sampling: true' to enable it."
                )
            _validate_domain_sampler(raw, context, integer=True)
            _validate_sampleable_integer(raw, context)
            return None
        if _is_param_ref(raw):
            return None
        value = int(raw)
        if float(value) != float(raw) or value < minimum:
            raise ValueError(f"{context} must be an integer >= {minimum}. Got {raw!r}.")
        return value

    def _float_param(name: str, *, positive: bool = False) -> float | None:
        return _float_value(
            params[name],
            f"{domain_type.capitalize()} domain parameter '{name}'",
            positive=positive,
        )

    def _vector_value(raw: Any, context: str, *, length: int) -> list[float] | None:
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{context} must be a list with {length} entries. Got {raw!r}."
            )
        values: list[float] = []
        saw_nonconcrete = False
        for index, value in enumerate(raw):
            entry_context = f"{context}[{index}]"
            if is_sampler_spec(value):
                if not allow_sampling:
                    raise ValueError(
                        f"{entry_context} uses sampled values, but domain sampling is "
                        "disabled. Set 'domain.allow_sampling: true' to enable it."
                    )
                _validate_domain_sampler(value, entry_context, integer=False)
                _validate_sampleable_numeric(value, entry_context)
                saw_nonconcrete = True
                continue
            if _is_param_ref(value):
                saw_nonconcrete = True
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{entry_context} must be finite. Got {numeric}.")
            values.append(numeric)
        if saw_nonconcrete:
            return None
        return values

    def _vector_param(name: str, *, length: int) -> list[float] | None:
        return _vector_value(
            params[name],
            f"{domain_type.capitalize()} domain parameter '{name}'",
            length=length,
        )

    def _positive_int_vector(name: str, *, length: int) -> list[int] | None:
        raw = params[name]
        if not isinstance(raw, list) or len(raw) != length:
            raise ValueError(
                f"{domain_type.capitalize()} domain parameter '{name}' must be a "
                f"list with {length} entries. Got {raw!r}."
            )
        values: list[int] = []
        saw_nonconcrete = False
        for index, value in enumerate(raw):
            context = f"{domain_type.capitalize()} domain parameter '{name}[{index}]'"
            if is_sampler_spec(value):
                if not allow_sampling:
                    raise ValueError(
                        f"{context} uses sampled values, but domain sampling is "
                        "disabled. Set 'domain.allow_sampling: true' to enable it."
                    )
                _validate_domain_sampler(value, context, integer=True)
                _validate_sampleable_integer(value, context)
                saw_nonconcrete = True
                continue
            if _is_param_ref(value):
                saw_nonconcrete = True
                continue
            int_value = int(value)
            if float(int_value) != float(value) or int_value <= 0:
                raise ValueError(
                    f"{context} must be a positive integer. Got {value!r}."
                )
            values.append(int_value)
        if saw_nonconcrete:
            return None
        return values

    if domain_type == "interval":
        _require_keys("size", "mesh_resolution")
        size = params["size"]
        if isinstance(size, list):
            if len(size) != 1:
                raise ValueError("Interval domain 'size' list must have length 1.")
            params["size"] = size
            _vector_param("size", length=1)
        else:
            _float_param("size", positive=True)
        res = params["mesh_resolution"]
        if isinstance(res, list):
            _positive_int_vector("mesh_resolution", length=1)
        else:
            if is_sampler_spec(res):
                if not allow_sampling:
                    raise ValueError(
                        "Interval domain parameter 'mesh_resolution' uses sampled "
                        "values, but domain sampling is disabled. Set "
                        "'domain.allow_sampling: true' to enable it."
                    )
                _validate_domain_sampler(
                    res,
                    "Interval domain parameter 'mesh_resolution'",
                    integer=True,
                )
                _validate_sampleable_integer(
                    res,
                    "Interval domain parameter 'mesh_resolution'",
                )
            else:
                int_value = int(res)
                if float(int_value) != float(res) or int_value <= 0:
                    raise ValueError(
                        "Interval domain parameter 'mesh_resolution' must be a "
                        f"positive integer. Got {res!r}."
                    )
        return

    if domain_type == "rectangle":
        _require_keys("size", "mesh_resolution")
        size = _vector_param("size", length=2)
        resolution = _positive_int_vector("mesh_resolution", length=2)
        if size is not None and any(value <= 0.0 for value in size):
            raise ValueError("Rectangle domain 'size' entries must be positive.")
        if resolution is not None and any(value <= 0 for value in resolution):
            raise ValueError(
                "Rectangle domain 'mesh_resolution' entries must be positive."
            )
        return

    if domain_type == "box":
        _require_keys("size", "mesh_resolution")
        size = _vector_param("size", length=3)
        resolution = _positive_int_vector("mesh_resolution", length=3)
        if size is not None and any(value <= 0.0 for value in size):
            raise ValueError("Box domain 'size' entries must be positive.")
        if resolution is not None and any(value <= 0 for value in resolution):
            raise ValueError("Box domain 'mesh_resolution' entries must be positive.")
        return

    if domain_type == "annulus":
        _require_keys("center", "inner_radius", "outer_radius", "mesh_size")
        _vector_param("center", length=2)
        inner_radius = _float_param("inner_radius", positive=True)
        outer_radius = _float_param("outer_radius", positive=True)
        _float_param("mesh_size", positive=True)
        if (
            inner_radius is not None
            and outer_radius is not None
            and inner_radius >= outer_radius
        ):
            raise ValueError(
                "Annulus domain requires 'inner_radius' < 'outer_radius'. "
                f"Got inner_radius={inner_radius} and outer_radius={outer_radius}."
            )
        return

    if domain_type == "disk":
        _require_keys("center", "radius", "mesh_size")
        _vector_param("center", length=2)
        _float_param("radius", positive=True)
        _float_param("mesh_size", positive=True)
        return

    if domain_type == "dumbbell":
        _require_keys(
            "left_center",
            "right_center",
            "lobe_radius",
            "neck_width",
            "mesh_size",
        )
        left_center = _vector_param("left_center", length=2)
        right_center = _vector_param("right_center", length=2)
        lobe_radius = _float_param("lobe_radius", positive=True)
        neck_width = _float_param("neck_width", positive=True)
        _float_param("mesh_size", positive=True)
        if (
            left_center is not None
            and right_center is not None
            and not math.isclose(left_center[1], right_center[1], abs_tol=1.0e-12)
        ):
            raise ValueError(
                "Dumbbell domain requires 'left_center' and 'right_center' to "
                "share the same y-coordinate."
            )
        if (
            left_center is not None
            and right_center is not None
            and right_center[0] <= left_center[0]
        ):
            raise ValueError(
                "Dumbbell domain requires 'right_center[0]' to be greater than "
                "'left_center[0]'."
            )
        if (
            neck_width is not None
            and lobe_radius is not None
            and neck_width > 2.0 * lobe_radius
        ):
            raise ValueError(
                "Dumbbell domain requires 'neck_width' <= 2 * 'lobe_radius'. "
                f"Got neck_width={neck_width} and lobe_radius={lobe_radius}."
            )
        return

    if domain_type == "l_shape":
        _require_keys(
            "outer_width",
            "outer_height",
            "cutout_width",
            "cutout_height",
            "mesh_size",
        )
        outer_width = _float_param("outer_width", positive=True)
        outer_height = _float_param("outer_height", positive=True)
        cutout_width = _float_param("cutout_width", positive=True)
        cutout_height = _float_param("cutout_height", positive=True)
        _float_param("mesh_size", positive=True)
        if (
            outer_width is not None
            and cutout_width is not None
            and cutout_width >= outer_width
        ):
            raise ValueError(
                "L_shape domain requires 'cutout_width' < 'outer_width'. "
                f"Got cutout_width={cutout_width} and outer_width={outer_width}."
            )
        if (
            outer_height is not None
            and cutout_height is not None
            and cutout_height >= outer_height
        ):
            raise ValueError(
                "L_shape domain requires 'cutout_height' < 'outer_height'. "
                f"Got cutout_height={cutout_height} and outer_height={outer_height}."
            )
        return

    if domain_type == "multi_hole_plate":
        _require_keys("width", "height", "holes", "mesh_size")
        width = _float_param("width", positive=True)
        height = _float_param("height", positive=True)
        _float_param("mesh_size", positive=True)
        holes_raw = params["holes"]
        if not isinstance(holes_raw, list) or not holes_raw:
            raise ValueError(
                "Multi_hole_plate domain parameter 'holes' must be a non-empty list."
            )

        concrete_holes: list[tuple[int, list[float], float]] = []
        for index, hole_raw in enumerate(holes_raw):
            context = f"Multi_hole_plate domain parameter 'holes[{index}]'"
            hole = _as_mapping(hole_raw, context)
            allowed_keys = {"center", "radius", "boundary_name"}
            if set(hole) - allowed_keys:
                raise ValueError(
                    f"{context} allows only {sorted(allowed_keys)}. Got {sorted(hole)}."
                )
            if not {"center", "radius"}.issubset(hole):
                raise ValueError(
                    f"{context} requires ['center', 'radius']. Got {sorted(hole)}."
                )

            center = _vector_value(hole["center"], f"{context}.center", length=2)
            radius = _float_value(hole["radius"], f"{context}.radius", positive=True)

            if "boundary_name" in hole:
                boundary_name = hole["boundary_name"]
                if not isinstance(boundary_name, str) or not boundary_name.strip():
                    raise ValueError(
                        f"{context}.boundary_name must be a non-empty string. "
                        f"Got {boundary_name!r}."
                    )
                if boundary_name == "outer":
                    raise ValueError(
                        f"{context}.boundary_name may not be 'outer' because that "
                        "name is reserved for the plate exterior."
                    )

            if center is not None and radius is not None:
                cx, cy = center
                if width is not None:
                    if cx - radius <= 0.0 or cx + radius >= width:
                        raise ValueError(
                            "Multi_hole_plate domain requires each hole to lie "
                            "strictly inside the plate in x."
                        )
                if height is not None:
                    if cy - radius <= 0.0 or cy + radius >= height:
                        raise ValueError(
                            "Multi_hole_plate domain requires each hole to lie "
                            "strictly inside the plate in y."
                        )
                concrete_holes.append((index, center, radius))

        for index, (hole_i, center_i, radius_i) in enumerate(concrete_holes):
            for hole_j, center_j, radius_j in concrete_holes[index + 1 :]:
                distance = math.hypot(
                    center_i[0] - center_j[0],
                    center_i[1] - center_j[1],
                )
                if distance <= radius_i + radius_j:
                    raise ValueError(
                        "Multi_hole_plate domain requires holes to remain "
                        f"non-overlapping. holes[{hole_i}] and holes[{hole_j}] "
                        "intersect."
                    )
        return

    if domain_type == "parallelogram":
        _require_keys("origin", "axis_x", "axis_y", "mesh_resolution")
        _vector_param("origin", length=2)
        axis_x = _vector_param("axis_x", length=2)
        axis_y = _vector_param("axis_y", length=2)
        _positive_int_vector("mesh_resolution", length=2)
        if axis_x is not None and axis_y is not None:
            signed_area = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
            if abs(signed_area) <= 1.0e-12:
                raise ValueError(
                    "Parallelogram domain requires linearly independent 'axis_x' and "
                    "'axis_y' vectors."
                )
        return

    if domain_type == "channel_obstacle":
        _require_keys(
            "length",
            "height",
            "obstacle_center",
            "obstacle_radius",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        height = _float_param("height", positive=True)
        obstacle_center = _vector_param("obstacle_center", length=2)
        obstacle_radius = _float_param("obstacle_radius", positive=True)
        _float_param("mesh_size", positive=True)
        if (
            length is not None
            and height is not None
            and obstacle_center is not None
            and obstacle_radius is not None
        ):
            cx, cy = obstacle_center
            if cx - obstacle_radius <= 0.0 or cx + obstacle_radius >= length:
                raise ValueError(
                    "Channel_obstacle domain requires the circular obstacle to lie "
                    "strictly inside the channel in x."
                )
            if cy - obstacle_radius <= 0.0 or cy + obstacle_radius >= height:
                raise ValueError(
                    "Channel_obstacle domain requires the circular obstacle to lie "
                    "strictly inside the channel in y."
                )
        return

    if domain_type == "y_bifurcation":
        _require_keys(
            "inlet_length",
            "branch_length",
            "branch_angle_degrees",
            "channel_width",
            "mesh_size",
        )
        inlet_length = _float_param("inlet_length", positive=True)
        branch_length = _float_param("branch_length", positive=True)
        branch_angle = _float_param("branch_angle_degrees", positive=True)
        channel_width = _float_param("channel_width", positive=True)
        _float_param("mesh_size", positive=True)
        if branch_angle is not None and branch_angle >= 85.0:
            raise ValueError(
                "Y_bifurcation domain requires 'branch_angle_degrees' < 85. "
                f"Got {branch_angle}."
            )
        if (
            inlet_length is not None
            and branch_length is not None
            and branch_angle is not None
            and channel_width is not None
        ):
            branch_angle_radians = math.radians(branch_angle)
            outlet_vertical_offset = branch_length * math.sin(branch_angle_radians)
            if outlet_vertical_offset <= channel_width:
                raise ValueError(
                    "Y_bifurcation domain requires "
                    "'branch_length * sin(branch_angle_degrees)' > 'channel_width' "
                    "so the two outlet branches separate cleanly."
                )
        return

    if domain_type == "venturi_channel":
        _require_keys(
            "length",
            "height",
            "throat_height",
            "constriction_center_x",
            "constriction_radius",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        height = _float_param("height", positive=True)
        throat_height = _float_param("throat_height", positive=True)
        constriction_center_x = _float_param("constriction_center_x", positive=True)
        constriction_radius = _float_param("constriction_radius", positive=True)
        _float_param("mesh_size", positive=True)
        if height is not None and throat_height is not None and throat_height >= height:
            raise ValueError(
                "Venturi_channel domain requires 'throat_height' < 'height'. "
                f"Got throat_height={throat_height} and height={height}."
            )
        if (
            height is not None
            and throat_height is not None
            and constriction_radius is not None
        ):
            indentation = 0.5 * (height - throat_height)
            if indentation >= constriction_radius:
                raise ValueError(
                    "Venturi_channel domain requires 'constriction_radius' to "
                    "exceed the wall indentation implied by "
                    "'height - throat_height'."
                )
        if (
            length is not None
            and height is not None
            and throat_height is not None
            and constriction_center_x is not None
            and constriction_radius is not None
        ):
            indentation = 0.5 * (height - throat_height)
            half_span = math.sqrt(
                max(
                    0.0,
                    2.0 * constriction_radius * indentation - indentation**2,
                )
            )
            if (
                constriction_center_x - half_span <= 0.0
                or constriction_center_x + half_span >= length
            ):
                raise ValueError(
                    "Venturi_channel domain requires the constriction to stay "
                    "strictly inside the channel in x."
                )
        return

    if domain_type == "porous_channel":
        _require_keys(
            "length",
            "height",
            "obstacle_radius",
            "n_rows",
            "n_cols",
            "pitch_x",
            "pitch_y",
            "x_margin",
            "y_margin",
            "row_shift_fraction",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        height = _float_param("height", positive=True)
        obstacle_radius = _float_param("obstacle_radius", positive=True)
        n_rows = _int_param("n_rows", minimum=1)
        n_cols = _int_param("n_cols", minimum=1)
        pitch_x = _float_param("pitch_x", positive=True)
        pitch_y = _float_param("pitch_y", positive=True)
        x_margin = _float_param("x_margin", positive=True)
        y_margin = _float_param("y_margin", positive=True)
        row_shift_fraction = _float_param("row_shift_fraction")
        _float_param("mesh_size", positive=True)

        if row_shift_fraction is not None and not 0.0 <= row_shift_fraction <= 0.5:
            raise ValueError(
                "Porous_channel domain requires 'row_shift_fraction' to lie in "
                f"[0, 0.5]. Got {row_shift_fraction}."
            )
        if (
            obstacle_radius is not None
            and pitch_x is not None
            and pitch_x <= 2.0 * obstacle_radius
        ):
            raise ValueError(
                "Porous_channel domain requires 'pitch_x' > 2 * "
                "'obstacle_radius' so neighboring obstacles do not overlap."
            )
        if (
            obstacle_radius is not None
            and pitch_y is not None
            and pitch_y <= 2.0 * obstacle_radius
        ):
            raise ValueError(
                "Porous_channel domain requires 'pitch_y' > 2 * "
                "'obstacle_radius' so stacked obstacle rows stay separated."
            )
        if (
            length is not None
            and obstacle_radius is not None
            and n_cols is not None
            and pitch_x is not None
            and x_margin is not None
            and row_shift_fraction is not None
        ):
            has_shifted_rows = n_rows is None or n_rows > 1
            shift_x = row_shift_fraction * pitch_x if has_shifted_rows else 0.0
            right_extent = (
                x_margin + 2.0 * obstacle_radius + (n_cols - 1) * pitch_x + shift_x
            )
            if right_extent >= length:
                raise ValueError(
                    "Porous_channel domain requires the obstacle array to lie "
                    "strictly inside the channel in x."
                )
        if (
            height is not None
            and obstacle_radius is not None
            and n_rows is not None
            and pitch_y is not None
            and y_margin is not None
        ):
            top_extent = y_margin + 2.0 * obstacle_radius + (n_rows - 1) * pitch_y
            if top_extent >= height:
                raise ValueError(
                    "Porous_channel domain requires the obstacle array to lie "
                    "strictly inside the channel in y."
                )
        return

    if domain_type == "serpentine_channel":
        _require_keys(
            "channel_length",
            "lane_spacing",
            "n_bends",
            "channel_width",
            "mesh_size",
        )
        channel_length = _float_param("channel_length", positive=True)
        lane_spacing = _float_param("lane_spacing", positive=True)
        _int_param("n_bends", minimum=2)
        channel_width = _float_param("channel_width", positive=True)
        _float_param("mesh_size", positive=True)
        if (
            channel_length is not None
            and channel_width is not None
            and channel_length <= channel_width
        ):
            raise ValueError(
                "Serpentine_channel domain requires 'channel_length' > "
                "'channel_width' so each lane has a non-degenerate straight span."
            )
        if (
            lane_spacing is not None
            and channel_width is not None
            and lane_spacing <= channel_width
        ):
            raise ValueError(
                "Serpentine_channel domain requires 'lane_spacing' > "
                "'channel_width' so adjacent lanes do not overlap."
            )
        return

    if domain_type == "airfoil_channel":
        _require_keys(
            "length",
            "height",
            "airfoil_center",
            "chord_length",
            "thickness_ratio",
            "attack_angle_degrees",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        height = _float_param("height", positive=True)
        airfoil_center = _vector_param("airfoil_center", length=2)
        chord_length = _float_param("chord_length", positive=True)
        thickness_ratio = _float_param("thickness_ratio", positive=True)
        attack_angle = _float_param("attack_angle_degrees")
        _float_param("mesh_size", positive=True)
        if thickness_ratio is not None and thickness_ratio >= 0.25:
            raise ValueError(
                "Airfoil_channel domain requires 'thickness_ratio' < 0.25. "
                f"Got {thickness_ratio}."
            )
        if attack_angle is not None and abs(attack_angle) >= 35.0:
            raise ValueError(
                "Airfoil_channel domain requires |'attack_angle_degrees'| < 35. "
                f"Got {attack_angle}."
            )
        if (
            length is not None
            and height is not None
            and airfoil_center is not None
            and chord_length is not None
            and thickness_ratio is not None
            and attack_angle is not None
        ):
            outline = symmetric_naca_airfoil_outline(
                chord_length=chord_length,
                thickness_ratio=thickness_ratio,
                center=np.asarray(airfoil_center, dtype=float),
                attack_angle_degrees=attack_angle,
            )
            x_min = float(np.min(outline[:, 0]))
            x_max = float(np.max(outline[:, 0]))
            y_min = float(np.min(outline[:, 1]))
            y_max = float(np.max(outline[:, 1]))
            if x_min <= 0.0 or x_max >= length:
                raise ValueError(
                    "Airfoil_channel domain requires the airfoil to lie strictly "
                    "inside the channel in x."
                )
            if y_min <= 0.0 or y_max >= height:
                raise ValueError(
                    "Airfoil_channel domain requires the airfoil to lie strictly "
                    "inside the channel in y."
                )
        return

    if domain_type == "side_cavity_channel":
        _require_keys(
            "length",
            "height",
            "cavity_width",
            "cavity_depth",
            "cavity_center_x",
            "mesh_size",
        )
        length = _float_param("length", positive=True)
        _float_param("height", positive=True)
        cavity_width = _float_param("cavity_width", positive=True)
        _float_param("cavity_depth", positive=True)
        cavity_center_x = _float_param("cavity_center_x", positive=True)
        _float_param("mesh_size", positive=True)
        if length is not None and cavity_width is not None and cavity_width >= length:
            raise ValueError(
                "Side_cavity_channel domain requires 'cavity_width' < 'length'. "
                f"Got cavity_width={cavity_width} and length={length}."
            )
        if (
            length is not None
            and cavity_width is not None
            and cavity_center_x is not None
        ):
            half_width = 0.5 * cavity_width
            if (
                cavity_center_x - half_width <= 0.0
                or cavity_center_x + half_width >= length
            ):
                raise ValueError(
                    "Side_cavity_channel domain requires the cavity to stay "
                    "strictly inside the channel in x."
                )
        return
