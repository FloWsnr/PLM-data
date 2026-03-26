"""Periodic-domain helpers backed by dolfinx_mpc."""

from collections.abc import Sequence
import importlib
from itertools import combinations
from typing import Any

import numpy as np
from dolfinx import fem

from plm_data.core.mesh import DomainGeometry

_AXIS_LABELS = ("x", "y", "z")


def periodic_boundary_names(domain_geom: DomainGeometry) -> set[str]:
    """Return boundary names that are unavailable due to periodicity."""
    blocked: set[str] = set()
    for axis in domain_geom.periodic_axes:
        label = _AXIS_LABELS[axis]
        blocked.add(f"{label}-")
        blocked.add(f"{label}+")
    return blocked


def require_dolfinx_mpc() -> Any:
    """Import dolfinx_mpc lazily with a clear installation error."""
    try:
        return importlib.import_module("dolfinx_mpc")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch tests
        raise RuntimeError(
            "Periodic domains require the optional 'dolfinx_mpc' package. "
            "Install it in the active DOLFINx environment before running a "
            "config with domain.periodic_axes."
        ) from exc


def require_unverified_periodic_support(
    preset_name: str,
    domain_geom: DomainGeometry,
    detail: str,
) -> None:
    """Fail fast when a periodic domain reaches an unverified space family."""
    if not domain_geom.has_periodic_axes:
        return
    raise NotImplementedError(
        f"Preset '{preset_name}' does not yet verify periodic domains for {detail}."
    )


def build_periodic_mpc(
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bcs: Sequence[fem.DirichletBC] | None = None,
    constrained_spaces: Sequence[fem.FunctionSpace] | None = None,
):
    """Build a finalized MPC for a space or selected subspaces."""
    if not domain_geom.has_periodic_axes:
        return None

    dolfinx_mpc = require_dolfinx_mpc()
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    spaces = [V] if constrained_spaces is None else list(constrained_spaces)
    dirichlet_bcs = [] if bcs is None else list(bcs)
    tol = _periodic_tol(domain_geom)

    for space in spaces:
        _add_periodic_constraints(
            mpc,
            space,
            domain_geom,
            dirichlet_bcs,
            tol,
        )

    mpc.finalize()
    return mpc


def _add_periodic_constraints(
    mpc,
    V: fem.FunctionSpace,
    domain_geom: DomainGeometry,
    bcs: Sequence[fem.DirichletBC],
    tol: float,
) -> None:
    """Attach all face/edge/corner periodic constraints for one space."""
    for periodic_subset in _periodic_axis_subsets(domain_geom.periodic_axes):
        anchor_axis = periodic_subset[-1]
        anchor_name = f"{_AXIS_LABELS[anchor_axis]}+"
        tag = domain_geom.boundary_names[anchor_name]
        relation = _build_periodic_relation(domain_geom, periodic_subset, tol)
        mpc.create_periodic_constraint_topological(
            V,
            domain_geom.facet_tags,
            tag,
            relation,
            list(bcs),
            tol=tol,
        )


def _periodic_axis_subsets(periodic_axes: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return non-empty periodic subsets ordered from faces to corners."""
    subsets: list[tuple[int, ...]] = []
    for size in range(1, len(periodic_axes) + 1):
        subsets.extend(combinations(periodic_axes, size))
    return subsets


def _build_periodic_relation(
    domain_geom: DomainGeometry,
    periodic_subset: tuple[int, ...],
    tol: float,
):
    """Build the slave-to-master map for one exact periodic subset."""
    periodic_axes = domain_geom.periodic_axes
    axis_lengths = {
        axis: domain_geom.axis_bounds[axis][1] - domain_geom.axis_bounds[axis][0]
        for axis in periodic_axes
    }
    positive_bounds = {axis: domain_geom.axis_bounds[axis][1] for axis in periodic_axes}

    def _relation(x):
        out = x.copy()
        mask = np.ones(x.shape[1], dtype=bool)

        for axis in periodic_axes:
            on_positive_face = np.isclose(
                x[axis], positive_bounds[axis], atol=tol, rtol=tol
            )
            if axis in periodic_subset:
                mask &= on_positive_face
            else:
                mask &= ~on_positive_face

        for axis in periodic_subset:
            out[axis, mask] = x[axis, mask] - axis_lengths[axis]

        out[:, ~mask] = np.nan
        return out

    return _relation


def _periodic_tol(domain_geom: DomainGeometry) -> float:
    """Return the MPC tolerance used for periodic matching."""
    return float(500 * np.finfo(domain_geom.mesh.geometry.x.dtype).eps)
