"""Periodic boundary helpers backed by dolfinx_mpc."""

from collections.abc import Sequence
import importlib
from itertools import combinations
from typing import Any

import numpy as np
from dolfinx import fem

from plm_data.core.config import BoundaryFieldConfig
from plm_data.core.mesh import DomainGeometry, PeriodicBoundaryMap


def periodic_boundary_names(boundary_field: BoundaryFieldConfig) -> set[str]:
    """Return boundary names that use the periodic operator."""
    blocked: set[str] = set()
    for name, entries in boundary_field.sides.items():
        if any(entry.type == "periodic" for entry in entries):
            blocked.add(name)
    return blocked


def require_dolfinx_mpc() -> Any:
    """Import dolfinx_mpc lazily with a clear installation error."""
    try:
        return importlib.import_module("dolfinx_mpc")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch tests
        raise RuntimeError(
            "Periodic boundary conditions require the optional 'dolfinx_mpc' "
            "package. Install it in the active DOLFINx environment before "
            "running a config with periodic boundary operators."
        ) from exc


def require_unverified_periodic_support(
    preset_name: str,
    boundary_field: BoundaryFieldConfig,
    detail: str,
) -> None:
    """Fail fast when a periodic field reaches an unverified space family."""
    if not boundary_field.has_periodic:
        return
    raise NotImplementedError(
        f"Preset '{preset_name}' does not yet verify periodic boundaries for {detail}."
    )


def build_periodic_mpc(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
    bcs: Sequence[fem.DirichletBC] | None = None,
    constrained_spaces: Sequence[fem.FunctionSpace] | None = None,
):
    """Build a finalized MPC for a space or selected subspaces."""
    active_maps = _active_periodic_maps(domain_geom, boundary_field)
    if not active_maps:
        return None

    dolfinx_mpc = require_dolfinx_mpc()
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    spaces = [V] if constrained_spaces is None else list(constrained_spaces)
    dirichlet_bcs = [] if bcs is None else list(bcs)
    tol = _periodic_tol(domain_geom)

    maps_by_group = _maps_by_group(active_maps)
    for space in spaces:
        _add_periodic_constraints(
            mpc,
            space,
            domain_geom,
            maps_by_group,
            dirichlet_bcs,
            tol,
        )

    mpc.finalize()
    return mpc


def _active_periodic_maps(
    domain_geom: DomainGeometry,
    boundary_field: BoundaryFieldConfig,
) -> list[PeriodicBoundaryMap]:
    """Resolve active periodic side pairs against the domain metadata."""
    active_maps = []
    for side_pair in sorted(
        boundary_field.periodic_pair_keys(),
        key=lambda pair: tuple(sorted(pair)),
    ):
        side_a, side_b = tuple(side_pair)
        active_maps.append(domain_geom.periodic_map(side_a, side_b))
    return active_maps


def _maps_by_group(
    periodic_maps: list[PeriodicBoundaryMap],
) -> dict[str, PeriodicBoundaryMap]:
    """Return active periodic maps keyed by composition group."""
    grouped: dict[str, PeriodicBoundaryMap] = {}
    for periodic_map in periodic_maps:
        if periodic_map.group_id in grouped:
            raise ValueError(
                f"Multiple active periodic maps share group id "
                f"'{periodic_map.group_id}'."
            )
        grouped[periodic_map.group_id] = periodic_map
    return grouped


def _add_periodic_constraints(
    mpc,
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    maps_by_group: dict[str, PeriodicBoundaryMap],
    bcs: Sequence[fem.DirichletBC],
    tol: float,
) -> None:
    """Attach all face/edge/corner periodic constraints for one space."""
    for periodic_subset in _periodic_group_subsets(tuple(sorted(maps_by_group))):
        anchor_group = periodic_subset[-1]
        anchor_map = maps_by_group[anchor_group]
        tag = domain_geom.boundary_names[anchor_map.slave_boundary]
        relation = _build_periodic_relation(maps_by_group, periodic_subset, tol)
        mpc.create_periodic_constraint_topological(
            V,
            domain_geom.facet_tags,
            tag,
            relation,
            list(bcs),
            tol=tol,
        )


def _periodic_group_subsets(group_ids: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Return non-empty periodic subsets ordered from faces to corners."""
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(group_ids) + 1):
        subsets.extend(combinations(group_ids, size))
    return subsets


def _build_periodic_relation(
    maps_by_group: dict[str, PeriodicBoundaryMap],
    periodic_subset: tuple[str, ...],
    tol: float,
):
    """Build the slave-to-master map for one exact periodic subset."""
    active_groups = tuple(sorted(maps_by_group))

    def _relation(x):
        out = x.copy()
        mask = np.ones(x.shape[1], dtype=bool)

        for group_id in active_groups:
            periodic_map = maps_by_group[group_id]
            on_slave = periodic_map.on_slave(x, tol)
            if group_id in periodic_subset:
                mask &= on_slave
            else:
                mask &= ~on_slave

        for group_id in periodic_subset:
            out[:, mask] = maps_by_group[group_id].apply(out[:, mask])

        out[:, ~mask] = np.nan
        return out

    return _relation


def _periodic_tol(domain_geom: DomainGeometry) -> float:
    """Return the MPC tolerance used for periodic matching."""
    return float(500 * np.finfo(domain_geom.mesh.geometry.x.dtype).eps)
