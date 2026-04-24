"""Gmsh-backed Y-bifurcation domain."""

import math

import numpy as np

from plm_data.domains.gmsh import register_gmsh_domain_factory
from plm_data.domains.helpers import require_param
from plm_data.domains.validation import DomainConfigLike, validate_domain_params


@register_gmsh_domain_factory("y_bifurcation", dimension=2)
def build_y_bifurcation_gmsh_model(model, domain: DomainConfigLike) -> None:
    """Populate the active Gmsh model with a tagged Y-bifurcation."""
    p = domain.params
    inlet_length = float(require_param(p, "inlet_length", domain.type))
    branch_length = float(require_param(p, "branch_length", domain.type))
    branch_angle_degrees = float(require_param(p, "branch_angle_degrees", domain.type))
    channel_width = float(require_param(p, "channel_width", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    branch_angle = math.radians(branch_angle_degrees)
    branch_axis = np.array([math.cos(branch_angle), math.sin(branch_angle)])
    junction_center = np.array([inlet_length, 0.0])
    inlet_center = np.array([0.0, 0.0])
    outlet_upper_center = junction_center + branch_length * branch_axis
    outlet_lower_center = junction_center + branch_length * np.array(
        [branch_axis[0], -branch_axis[1]]
    )
    cap_tolerance = 0.45 * channel_width + 1.0e-8

    half_width = 0.5 * channel_width
    trunk = model.occ.addRectangle(
        0.0,
        -half_width,
        0.0,
        inlet_length,
        channel_width,
    )
    upper_branch = model.occ.addRectangle(
        inlet_length,
        -half_width,
        0.0,
        branch_length,
        channel_width,
    )
    lower_branch = model.occ.addRectangle(
        inlet_length,
        -half_width,
        0.0,
        branch_length,
        channel_width,
    )
    model.occ.rotate(
        [(2, upper_branch)],
        inlet_length,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        branch_angle,
    )
    model.occ.rotate(
        [(2, lower_branch)],
        inlet_length,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        -branch_angle,
    )
    junction_disk = model.occ.addDisk(
        inlet_length,
        0.0,
        0.0,
        half_width,
        half_width,
    )
    model.occ.fuse(
        [(2, trunk)],
        [
            (2, upper_branch),
            (2, lower_branch),
            (2, junction_disk),
        ],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    inlet: list[int] = []
    outlet_upper: list[int] = []
    outlet_lower: list[int] = []
    walls: list[int] = []
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        center = np.asarray(
            model.occ.getCenterOfMass(dim, tag)[:2],
            dtype=float,
        )
        if np.linalg.norm(center - inlet_center) <= cap_tolerance:
            inlet.append(tag)
            continue
        if np.linalg.norm(center - outlet_upper_center) <= cap_tolerance:
            outlet_upper.append(tag)
            continue
        if np.linalg.norm(center - outlet_lower_center) <= cap_tolerance:
            outlet_lower.append(tag)
            continue
        walls.append(tag)

    for physical_tag, (name, curve_tags) in enumerate(
        (
            ("inlet", inlet),
            ("outlet_upper", outlet_upper),
            ("outlet_lower", outlet_lower),
            ("walls", walls),
        ),
        start=1,
    ):
        if not curve_tags:
            raise AssertionError(
                f"Y-bifurcation domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)
