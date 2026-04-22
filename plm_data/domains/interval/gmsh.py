"""Gmsh-backed interval domain."""

import numpy as np

from plm_data.core.config import DomainConfig, validate_domain_params
from plm_data.domains.base import DomainGeometry, register_domain, register_gmsh_domain
from plm_data.domains.gmsh import create_gmsh_domain
from plm_data.domains.helpers import (
    builtin_periodic_map,
    compile_config_periodic_maps,
    merge_periodic_maps,
    require_param,
)


def _length_and_resolution(domain: DomainConfig) -> tuple[float, int]:
    p = domain.params
    size = require_param(p, "size", domain.type)
    res = require_param(p, "mesh_resolution", domain.type)
    length = float(size[0] if isinstance(size, list) else size)
    nx = int(res[0] if isinstance(res, list) else res)
    return length, nx


@register_gmsh_domain("interval", dimension=1)
def build_interval_gmsh_model(model, domain: DomainConfig) -> None:
    """Populate the active Gmsh model with a tagged interval."""
    validate_domain_params(domain.type, domain.params)
    length, nx = _length_and_resolution(domain)

    x_min = model.occ.addPoint(0.0, 0.0, 0.0)
    x_max = model.occ.addPoint(length, 0.0, 0.0)
    line = model.occ.addLine(x_min, x_max)
    model.occ.synchronize()

    model.addPhysicalGroup(1, [line], tag=1)
    model.setPhysicalName(1, 1, "line")
    model.addPhysicalGroup(0, [x_min], tag=1)
    model.setPhysicalName(0, 1, "x-")
    model.addPhysicalGroup(0, [x_max], tag=2)
    model.setPhysicalName(0, 2, "x+")

    model.mesh.setTransfiniteCurve(line, nx + 1)


@register_domain("interval")
def create_interval(domain: DomainConfig) -> DomainGeometry:
    """Create a tagged 1D interval through Gmsh."""
    length, _ = _length_and_resolution(domain)
    domain_geom = create_gmsh_domain(domain)
    builtin_maps = builtin_periodic_map(
        name="x",
        group_id="x",
        slave_boundary="x+",
        master_boundary="x-",
        slave_selector=lambda x, tol, lim=length: np.isclose(
            x[0],
            lim,
            atol=tol,
            rtol=tol,
        ),
        offset=(-length,),
    )
    domain_geom.periodic_maps = merge_periodic_maps(
        builtin_maps,
        compile_config_periodic_maps(domain.periodic_maps),
    )
    return domain_geom
