"""Sampling and validation metadata for the porous-channel domain."""

from plm_data.domains.base import (
    COMMON_BOUNDARY_FAMILIES,
    COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
    DomainParameterSpec,
    DomainSpec,
    register_domain_spec,
)


DOMAIN_SPEC = register_domain_spec(
    DomainSpec(
        name="porous_channel",
        dimension=2,
        description="Rectangular channel with a staggered array of circular obstacles.",
        parameters={
            "length": DomainParameterSpec(
                name="length",
                kind="float",
                hard_min=0.0,
                sampling_min=1.5,
                sampling_max=4.0,
                description="Channel length.",
            ),
            "height": DomainParameterSpec(
                name="height",
                kind="float",
                hard_min=0.0,
                sampling_min=0.8,
                sampling_max=1.8,
                description="Channel height.",
            ),
            "obstacle_radius": DomainParameterSpec(
                name="obstacle_radius",
                kind="float",
                hard_min=0.0,
                sampling_min=0.04,
                sampling_max=0.16,
                description="Radius of each obstacle.",
            ),
            "n_rows": DomainParameterSpec(
                name="n_rows",
                kind="int",
                hard_min=1,
                sampling_min=2,
                sampling_max=6,
                description="Number of obstacle rows.",
            ),
            "n_cols": DomainParameterSpec(
                name="n_cols",
                kind="int",
                hard_min=1,
                sampling_min=2,
                sampling_max=8,
                description="Number of obstacle columns.",
            ),
            "pitch_x": DomainParameterSpec(
                name="pitch_x",
                kind="float",
                hard_min=0.0,
                description="Horizontal obstacle spacing.",
            ),
            "pitch_y": DomainParameterSpec(
                name="pitch_y",
                kind="float",
                hard_min=0.0,
                description="Vertical obstacle spacing.",
            ),
            "x_margin": DomainParameterSpec(
                name="x_margin",
                kind="float",
                hard_min=0.0,
                description="Left obstacle-array margin.",
            ),
            "y_margin": DomainParameterSpec(
                name="y_margin",
                kind="float",
                hard_min=0.0,
                description="Lower obstacle-array margin.",
            ),
            "row_shift_fraction": DomainParameterSpec(
                name="row_shift_fraction",
                kind="float",
                hard_min=0.0,
                hard_max=0.5,
                sampling_min=0.0,
                sampling_max=0.5,
                description="Fractional x-shift applied to odd rows.",
            ),
            "mesh_size": DomainParameterSpec(
                name="mesh_size",
                kind="float",
                hard_min=0.0,
                sampling_min=0.03,
                sampling_max=0.14,
                description="Target Gmsh mesh size.",
            ),
        },
        boundary_names=("inlet", "outlet", "walls", "obstacles"),
        boundary_roles={
            "all": ("inlet", "outlet", "walls", "obstacles"),
            "inlet": ("inlet",),
            "outlet": ("outlet",),
            "walls": ("walls",),
            "obstacles": ("obstacles",),
            "solid": ("walls", "obstacles"),
            "open": ("inlet", "outlet"),
        },
        allowed_boundary_families=COMMON_BOUNDARY_FAMILIES
        + ("open_channel", "porous_obstacle_drive", "inlet_outlet_drive"),
        allowed_initial_condition_families=COMMON_SCALAR_INITIAL_CONDITION_FAMILIES,
        coordinate_regions=(
            "interior",
            "pore_space",
            "upstream",
            "downstream",
            "near_obstacles",
        ),
    )
)
