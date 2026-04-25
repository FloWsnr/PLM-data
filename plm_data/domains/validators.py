"""Domain-specific parameter validators referenced by domain specs."""

import math
from plm_data.domains.validation import DomainValidationContext


def validate_rectangle_params(ctx: DomainValidationContext) -> None:
    size = ctx.vector_param("size", length=2)
    resolution = ctx.positive_int_vector("mesh_resolution", length=2)
    if size is not None and any(value <= 0.0 for value in size):
        raise ValueError("Rectangle domain 'size' entries must be positive.")
    if resolution is not None and any(value <= 0 for value in resolution):
        raise ValueError("Rectangle domain 'mesh_resolution' entries must be positive.")


def validate_annulus_params(ctx: DomainValidationContext) -> None:
    ctx.vector_param("center", length=2)
    inner_radius = ctx.float_param("inner_radius", positive=True)
    outer_radius = ctx.float_param("outer_radius", positive=True)
    ctx.float_param("mesh_size", positive=True)
    if (
        inner_radius is not None
        and outer_radius is not None
        and inner_radius >= outer_radius
    ):
        raise ValueError(
            "Annulus domain requires 'inner_radius' < 'outer_radius'. "
            f"Got inner_radius={inner_radius} and outer_radius={outer_radius}."
        )


def validate_disk_params(ctx: DomainValidationContext) -> None:
    ctx.vector_param("center", length=2)
    ctx.float_param("radius", positive=True)
    ctx.float_param("mesh_size", positive=True)


def validate_dumbbell_params(ctx: DomainValidationContext) -> None:
    left_center = ctx.vector_param("left_center", length=2)
    right_center = ctx.vector_param("right_center", length=2)
    lobe_radius = ctx.float_param("lobe_radius", positive=True)
    neck_width = ctx.float_param("neck_width", positive=True)
    ctx.float_param("mesh_size", positive=True)
    if (
        left_center is not None
        and right_center is not None
        and not math.isclose(left_center[1], right_center[1], abs_tol=1.0e-12)
    ):
        raise ValueError(
            "Dumbbell domain requires 'left_center' and 'right_center' to share the "
            "same y-coordinate."
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


def validate_l_shape_params(ctx: DomainValidationContext) -> None:
    outer_width = ctx.float_param("outer_width", positive=True)
    outer_height = ctx.float_param("outer_height", positive=True)
    cutout_width = ctx.float_param("cutout_width", positive=True)
    cutout_height = ctx.float_param("cutout_height", positive=True)
    ctx.float_param("mesh_size", positive=True)
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


def validate_multi_hole_plate_params(ctx: DomainValidationContext) -> None:
    width = ctx.float_param("width", positive=True)
    height = ctx.float_param("height", positive=True)
    ctx.float_param("mesh_size", positive=True)
    holes_raw = ctx.params["holes"]
    if not isinstance(holes_raw, list) or not holes_raw:
        raise ValueError(
            "Multi_hole_plate domain parameter 'holes' must be a non-empty list."
        )

    concrete_holes: list[tuple[int, list[float], float]] = []
    for index, hole_raw in enumerate(holes_raw):
        context = f"Multi_hole_plate domain parameter 'holes[{index}]'"
        hole = ctx.as_mapping(hole_raw, context)
        allowed_keys = {"center", "radius", "boundary_name"}
        if set(hole) - allowed_keys:
            raise ValueError(
                f"{context} allows only {sorted(allowed_keys)}. Got {sorted(hole)}."
            )
        if not {"center", "radius"}.issubset(hole):
            raise ValueError(
                f"{context} requires ['center', 'radius']. Got {sorted(hole)}."
            )

        center = ctx.vector_value(hole["center"], f"{context}.center", length=2)
        radius = ctx.float_value(hole["radius"], f"{context}.radius", positive=True)

        if "boundary_name" in hole:
            boundary_name = hole["boundary_name"]
            if not isinstance(boundary_name, str) or not boundary_name.strip():
                raise ValueError(
                    f"{context}.boundary_name must be a non-empty string. "
                    f"Got {boundary_name!r}."
                )
            if boundary_name == "outer":
                raise ValueError(
                    f"{context}.boundary_name may not be 'outer' because that name "
                    "is reserved for the plate exterior."
                )

        if center is not None and radius is not None:
            cx, cy = center
            if width is not None and (cx - radius <= 0.0 or cx + radius >= width):
                raise ValueError(
                    "Multi_hole_plate domain requires each hole to lie strictly "
                    "inside the plate in x."
                )
            if height is not None and (cy - radius <= 0.0 or cy + radius >= height):
                raise ValueError(
                    "Multi_hole_plate domain requires each hole to lie strictly "
                    "inside the plate in y."
                )
            concrete_holes.append((index, center, radius))

    for index, (hole_i, center_i, radius_i) in enumerate(concrete_holes):
        for hole_j, center_j, radius_j in concrete_holes[index + 1 :]:
            distance = math.hypot(center_i[0] - center_j[0], center_i[1] - center_j[1])
            if distance <= radius_i + radius_j:
                raise ValueError(
                    "Multi_hole_plate domain requires holes to remain non-overlapping. "
                    f"holes[{hole_i}] and holes[{hole_j}] intersect."
                )


def validate_parallelogram_params(ctx: DomainValidationContext) -> None:
    ctx.vector_param("origin", length=2)
    axis_x = ctx.vector_param("axis_x", length=2)
    axis_y = ctx.vector_param("axis_y", length=2)
    ctx.positive_int_vector("mesh_resolution", length=2)
    if axis_x is not None and axis_y is not None:
        signed_area = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0]
        if abs(signed_area) <= 1.0e-12:
            raise ValueError(
                "Parallelogram domain requires linearly independent 'axis_x' and "
                "'axis_y' vectors."
            )


def validate_channel_obstacle_params(ctx: DomainValidationContext) -> None:
    length = ctx.float_param("length", positive=True)
    height = ctx.float_param("height", positive=True)
    obstacle_center = ctx.vector_param("obstacle_center", length=2)
    obstacle_radius = ctx.float_param("obstacle_radius", positive=True)
    ctx.float_param("mesh_size", positive=True)
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


def validate_y_bifurcation_params(ctx: DomainValidationContext) -> None:
    inlet_length = ctx.float_param("inlet_length", positive=True)
    branch_length = ctx.float_param("branch_length", positive=True)
    branch_angle = ctx.float_param("branch_angle_degrees", positive=True)
    channel_width = ctx.float_param("channel_width", positive=True)
    ctx.float_param("mesh_size", positive=True)
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
        outlet_vertical_offset = branch_length * math.sin(math.radians(branch_angle))
        if outlet_vertical_offset <= channel_width:
            raise ValueError(
                "Y_bifurcation domain requires "
                "'branch_length * sin(branch_angle_degrees)' > 'channel_width' "
                "so the two outlet branches separate cleanly."
            )


def validate_venturi_channel_params(ctx: DomainValidationContext) -> None:
    length = ctx.float_param("length", positive=True)
    height = ctx.float_param("height", positive=True)
    throat_height = ctx.float_param("throat_height", positive=True)
    constriction_center_x = ctx.float_param("constriction_center_x", positive=True)
    constriction_radius = ctx.float_param("constriction_radius", positive=True)
    ctx.float_param("mesh_size", positive=True)
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
                "Venturi_channel domain requires 'constriction_radius' to exceed the "
                "wall indentation implied by 'height - throat_height'."
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
            max(0.0, 2.0 * constriction_radius * indentation - indentation**2)
        )
        if (
            constriction_center_x - half_span <= 0.0
            or constriction_center_x + half_span >= length
        ):
            raise ValueError(
                "Venturi_channel domain requires the constriction to stay strictly "
                "inside the channel in x."
            )


def validate_porous_channel_params(ctx: DomainValidationContext) -> None:
    length = ctx.float_param("length", positive=True)
    height = ctx.float_param("height", positive=True)
    obstacle_radius = ctx.float_param("obstacle_radius", positive=True)
    n_rows = ctx.int_param("n_rows", minimum=1)
    n_cols = ctx.int_param("n_cols", minimum=1)
    pitch_x = ctx.float_param("pitch_x", positive=True)
    pitch_y = ctx.float_param("pitch_y", positive=True)
    x_margin = ctx.float_param("x_margin", positive=True)
    y_margin = ctx.float_param("y_margin", positive=True)
    row_shift_fraction = ctx.float_param("row_shift_fraction")
    ctx.float_param("mesh_size", positive=True)

    if row_shift_fraction is not None and not 0.0 <= row_shift_fraction <= 0.5:
        raise ValueError(
            "Porous_channel domain requires 'row_shift_fraction' to lie in [0, 0.5]. "
            f"Got {row_shift_fraction}."
        )
    if (
        obstacle_radius is not None
        and pitch_x is not None
        and pitch_x <= 2.0 * obstacle_radius
    ):
        raise ValueError(
            "Porous_channel domain requires 'pitch_x' > 2 * 'obstacle_radius' so "
            "neighboring obstacles do not overlap."
        )
    if (
        obstacle_radius is not None
        and pitch_y is not None
        and pitch_y <= 2.0 * obstacle_radius
    ):
        raise ValueError(
            "Porous_channel domain requires 'pitch_y' > 2 * 'obstacle_radius' so "
            "stacked obstacle rows stay separated."
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
                "Porous_channel domain requires the obstacle array to lie strictly "
                "inside the channel in x."
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
                "Porous_channel domain requires the obstacle array to lie strictly "
                "inside the channel in y."
            )


def validate_serpentine_channel_params(ctx: DomainValidationContext) -> None:
    channel_length = ctx.float_param("channel_length", positive=True)
    lane_spacing = ctx.float_param("lane_spacing", positive=True)
    ctx.int_param("n_bends", minimum=2)
    channel_width = ctx.float_param("channel_width", positive=True)
    ctx.float_param("mesh_size", positive=True)
    if (
        channel_length is not None
        and channel_width is not None
        and channel_length <= channel_width
    ):
        raise ValueError(
            "Serpentine_channel domain requires 'channel_length' > 'channel_width' "
            "so each lane has a non-degenerate straight span."
        )
    if (
        lane_spacing is not None
        and channel_width is not None
        and lane_spacing <= channel_width
    ):
        raise ValueError(
            "Serpentine_channel domain requires 'lane_spacing' > 'channel_width' "
            "so adjacent lanes do not overlap."
        )


def validate_side_cavity_channel_params(ctx: DomainValidationContext) -> None:
    length = ctx.float_param("length", positive=True)
    ctx.float_param("height", positive=True)
    cavity_width = ctx.float_param("cavity_width", positive=True)
    ctx.float_param("cavity_depth", positive=True)
    cavity_center_x = ctx.float_param("cavity_center_x", positive=True)
    ctx.float_param("mesh_size", positive=True)
    if length is not None and cavity_width is not None and cavity_width >= length:
        raise ValueError(
            "Side_cavity_channel domain requires 'cavity_width' < 'length'. "
            f"Got cavity_width={cavity_width} and length={length}."
        )
    if length is not None and cavity_width is not None and cavity_center_x is not None:
        half_width = 0.5 * cavity_width
        if (
            cavity_center_x - half_width <= 0.0
            or cavity_center_x + half_width >= length
        ):
            raise ValueError(
                "Side_cavity_channel domain requires the cavity to stay strictly "
                "inside the channel in x."
            )
