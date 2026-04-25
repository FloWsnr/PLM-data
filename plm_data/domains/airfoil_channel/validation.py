"""Parameter validation for the airfoil-channel domain."""

import numpy as np

from plm_data.domains.airfoil_channel.profile import symmetric_naca_airfoil_outline
from plm_data.domains.validation import DomainValidationContext


def validate_airfoil_channel_params(ctx: DomainValidationContext) -> None:
    length = ctx.float_param("length", positive=True)
    height = ctx.float_param("height", positive=True)
    airfoil_center = ctx.vector_param("airfoil_center", length=2)
    chord_length = ctx.float_param("chord_length", positive=True)
    thickness_ratio = ctx.float_param("thickness_ratio", positive=True)
    attack_angle = ctx.float_param("attack_angle_degrees")
    ctx.float_param("mesh_size", positive=True)
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
                "Airfoil_channel domain requires the airfoil to lie strictly inside "
                "the channel in x."
            )
        if y_min <= 0.0 or y_max >= height:
            raise ValueError(
                "Airfoil_channel domain requires the airfoil to lie strictly inside "
                "the channel in y."
            )
