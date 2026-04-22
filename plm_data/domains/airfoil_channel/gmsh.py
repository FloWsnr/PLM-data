"""Gmsh-backed airfoil channel domain."""

import numpy as np

from plm_data.core.airfoil import symmetric_naca_airfoil_surfaces
from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.gmsh import register_gmsh_domain_factory
from plm_data.domains.helpers import require_param


@register_gmsh_domain_factory("airfoil_channel", dimension=2)
def build_airfoil_channel_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged airfoil channel."""
    p = domain.params
    length = float(require_param(p, "length", domain.type))
    height = float(require_param(p, "height", domain.type))
    airfoil_center = np.asarray(
        require_param(p, "airfoil_center", domain.type),
        dtype=float,
    )
    chord_length = float(require_param(p, "chord_length", domain.type))
    thickness_ratio = float(require_param(p, "thickness_ratio", domain.type))
    attack_angle_degrees = float(require_param(p, "attack_angle_degrees", domain.type))
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    upper_points, lower_points = symmetric_naca_airfoil_surfaces(
        chord_length=chord_length,
        thickness_ratio=thickness_ratio,
        center=airfoil_center,
        attack_angle_degrees=attack_angle_degrees,
    )

    channel = model.occ.addRectangle(0.0, 0.0, 0.0, length, height)

    upper_tags = [
        model.occ.addPoint(float(x_coord), float(y_coord), 0.0)
        for x_coord, y_coord in upper_points
    ]
    leading_edge_tag = upper_tags[-1]
    lower_tags = [leading_edge_tag] + [
        model.occ.addPoint(float(x_coord), float(y_coord), 0.0)
        for x_coord, y_coord in lower_points
    ]
    upper_spline = model.occ.addSpline(upper_tags)
    lower_spline = model.occ.addSpline(lower_tags)
    trailing_edge = model.occ.addLine(lower_tags[-1], upper_tags[0])
    airfoil_loop = model.occ.addCurveLoop([upper_spline, lower_spline, trailing_edge])
    airfoil_surface = model.occ.addPlaneSurface([airfoil_loop])

    model.occ.cut(
        [(2, channel)],
        [(2, airfoil_surface)],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    tol = max(mesh_size, 1.0e-8)
    inlet: list[int] = []
    outlet: list[int] = []
    walls: list[int] = []
    airfoil_curves: list[int] = []
    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        bb = model.occ.getBoundingBox(dim, tag)
        x_min, y_min, _, x_max, y_max, _ = bb
        if np.isclose(x_min, 0.0, atol=tol) and np.isclose(
            x_max,
            0.0,
            atol=tol,
        ):
            inlet.append(tag)
        elif np.isclose(x_min, length, atol=tol) and np.isclose(
            x_max,
            length,
            atol=tol,
        ):
            outlet.append(tag)
        elif np.isclose(y_min, 0.0, atol=tol) or np.isclose(
            y_max,
            height,
            atol=tol,
        ):
            walls.append(tag)
        else:
            airfoil_curves.append(tag)

    for physical_tag, (name, curve_tags) in enumerate(
        (
            ("inlet", inlet),
            ("outlet", outlet),
            ("walls", walls),
            ("airfoil", airfoil_curves),
        ),
        start=1,
    ):
        if not curve_tags:
            raise AssertionError(
                f"Airfoil-channel domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)
