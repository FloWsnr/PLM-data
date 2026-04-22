"""Reusable mesh-construction helpers for concrete domain packages."""

from collections.abc import Callable
from typing import Any

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.mesh import locate_entities_boundary, meshtags

from plm_data.domains.base import PeriodicBoundaryMap


def require_param(params: dict[str, Any], key: str, domain_type: str) -> Any:
    """Require a domain parameter, raising a clear error if missing."""
    if key not in params:
        raise ValueError(
            f"Missing required parameter '{key}' for domain type '{domain_type}'. "
            f"Got params: {params}"
        )
    return params[key]


def tag_boundaries(
    msh: mesh.Mesh,
    predicates: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> tuple[mesh.MeshTags, dict[str, int], ufl.Measure]:  # type: ignore[reportInvalidTypeForm]
    """Tag boundary facets using geometric predicates."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)

    boundary_names: dict[str, int] = {}
    all_facets = []
    all_tags = []

    for tag_idx, (name, marker) in enumerate(predicates.items(), start=1):
        boundary_names[name] = tag_idx
        facets = locate_entities_boundary(msh, fdim, marker)
        all_facets.append(facets)
        all_tags.append(np.full_like(facets, tag_idx))

    if all_facets:
        facet_indices = np.concatenate(all_facets)
        tag_values = np.concatenate(all_tags)
        sort_order = np.argsort(facet_indices)
        facet_indices = facet_indices[sort_order]
        tag_values = tag_values[sort_order]
    else:
        facet_indices = np.empty(0, dtype=np.int32)
        tag_values = np.empty(0, dtype=np.int32)

    ft = meshtags(msh, fdim, facet_indices, tag_values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    return ft, boundary_names, ds


def compile_config_periodic_maps(
    periodic_maps: dict[str, Any],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Compile declarative domain periodic maps into runtime maps."""
    compiled: dict[frozenset[str], PeriodicBoundaryMap] = {}
    for name, map_config in periodic_maps.items():
        key = frozenset({map_config.slave, map_config.master})
        if key in compiled:
            raise ValueError(
                f"Duplicate periodic domain map for boundary pair {sorted(key)}."
            )
        compiled[key] = PeriodicBoundaryMap(
            name=name,
            slave_boundary=map_config.slave,
            master_boundary=map_config.master,
            matrix=np.asarray(map_config.matrix, dtype=float),
            offset=np.asarray(map_config.offset, dtype=float),
            group_id=name,
        )
    return compiled


def merge_periodic_maps(
    *maps: dict[frozenset[str], PeriodicBoundaryMap],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Merge periodic map dictionaries, rejecting duplicate side pairs."""
    merged: dict[frozenset[str], PeriodicBoundaryMap] = {}
    for periodic_maps in maps:
        for key, periodic_map in periodic_maps.items():
            if key in merged:
                raise ValueError(
                    f"Duplicate periodic map for boundary pair {sorted(key)}."
                )
            merged[key] = periodic_map
    return merged


def builtin_periodic_map(
    *,
    name: str,
    group_id: str,
    slave_boundary: str,
    master_boundary: str,
    slave_selector: Callable[[np.ndarray, float], np.ndarray],
    offset: tuple[float, ...],
) -> dict[frozenset[str], PeriodicBoundaryMap]:
    """Build one built-in affine periodic pair map."""
    gdim = len(offset)
    return {
        frozenset({slave_boundary, master_boundary}): PeriodicBoundaryMap(
            name=name,
            slave_boundary=slave_boundary,
            master_boundary=master_boundary,
            matrix=np.eye(gdim, dtype=float),
            offset=np.asarray(offset, dtype=float),
            group_id=group_id,
            slave_selector=slave_selector,
        )
    }
