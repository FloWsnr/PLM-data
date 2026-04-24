"""Gmsh-backed multi-hole plate domain."""

import numpy as np

from plm_data.domains.gmsh import register_gmsh_domain_factory
from plm_data.domains.helpers import require_param
from plm_data.domains.validation import DomainConfigLike, validate_domain_params


@register_gmsh_domain_factory("multi_hole_plate", dimension=2)
def build_multi_hole_plate_gmsh_model(model, domain: DomainConfigLike) -> None:
    """Populate the active Gmsh model with a tagged multi-hole plate."""
    p = domain.params
    width = float(require_param(p, "width", domain.type))
    height = float(require_param(p, "height", domain.type))
    holes_raw = require_param(p, "holes", domain.type)
    mesh_size = float(require_param(p, "mesh_size", domain.type))
    validate_domain_params(domain.type, p)

    if not isinstance(holes_raw, list):
        raise AssertionError("multi_hole_plate requires 'holes' to be a list.")

    hole_specs: list[tuple[np.ndarray, float, str]] = []
    for hole_raw in holes_raw:
        if not isinstance(hole_raw, dict):
            raise AssertionError(
                "multi_hole_plate hole definitions must be mappings after realization."
            )
        hole_specs.append(
            (
                np.asarray(hole_raw["center"], dtype=float),
                float(hole_raw["radius"]),
                str(hole_raw.get("boundary_name", "holes")),
            )
        )

    plate = model.occ.addRectangle(0.0, 0.0, 0.0, width, height)
    holes = [
        model.occ.addDisk(center[0], center[1], 0.0, radius, radius)
        for center, radius, _ in hole_specs
    ]
    model.occ.cut(
        [(2, plate)],
        [(2, hole) for hole in holes],
        removeObject=True,
        removeTool=True,
    )
    model.occ.synchronize()

    surfaces = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surfaces, tag=1)
    model.setPhysicalName(2, 1, "surface")

    tol = max(1.0e-8, 1.0e-6 * max(width, height))
    outer: list[int] = []
    hole_groups: dict[str, list[int]] = {}
    for _, _, boundary_name in hole_specs:
        hole_groups.setdefault(boundary_name, [])

    boundary = model.getBoundary(
        [(2, surface_tag) for surface_tag in surfaces],
        oriented=False,
    )
    for dim, tag in boundary:
        if dim != 1:
            continue
        x_min, y_min, _, x_max, y_max, _ = model.occ.getBoundingBox(dim, tag)
        if (
            (np.isclose(x_min, 0.0, atol=tol) and np.isclose(x_max, 0.0, atol=tol))
            or (
                np.isclose(x_min, width, atol=tol)
                and np.isclose(x_max, width, atol=tol)
            )
            or (np.isclose(y_min, 0.0, atol=tol) and np.isclose(y_max, 0.0, atol=tol))
            or (
                np.isclose(y_min, height, atol=tol)
                and np.isclose(y_max, height, atol=tol)
            )
        ):
            outer.append(tag)
            continue

        curve_center = np.asarray(
            model.occ.getCenterOfMass(dim, tag)[:2],
            dtype=float,
        )
        hole_index = min(
            range(len(hole_specs)),
            key=lambda index: np.linalg.norm(curve_center - hole_specs[index][0]),
        )
        hole_groups[hole_specs[hole_index][2]].append(tag)

    physical_groups = [("outer", outer), *hole_groups.items()]
    for physical_tag, (name, curve_tags) in enumerate(
        physical_groups,
        start=1,
    ):
        if not curve_tags:
            raise AssertionError(
                f"Multi-hole plate domain produced no curves for '{name}'."
            )
        model.addPhysicalGroup(1, curve_tags, tag=physical_tag)
        model.setPhysicalName(1, physical_tag, name)

    model.mesh.setSize(model.getEntities(0), mesh_size)
