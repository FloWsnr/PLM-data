"""Shared PDE-level boundary-condition validation helpers."""

from plm_data.core.runtime_config import BoundaryFieldConfig
from plm_data.core.mesh import DomainGeometry


def validate_boundary_field_structure(
    *,
    pde_name: str,
    field_name: str,
    boundary_field: BoundaryFieldConfig,
    domain_geom: DomainGeometry,
    allowed_operators: set[str],
    expected_entries_per_side: int = 1,
) -> None:
    """Validate side coverage, side names, operator names, and periodic maps."""
    expected_sides = set(domain_geom.boundary_names)
    actual_sides = set(boundary_field.sides)
    if actual_sides != expected_sides:
        raise ValueError(
            f"PDE '{pde_name}' boundary field '{field_name}' must configure "
            f"exactly the domain sides {sorted(expected_sides)}. Got "
            f"{sorted(actual_sides)}."
        )

    for side_name, entries in boundary_field.sides.items():
        if len(entries) != expected_entries_per_side:
            raise ValueError(
                f"PDE '{pde_name}' boundary field '{field_name}' side "
                f"'{side_name}' must have exactly {expected_entries_per_side} "
                f"operator entry/entries. Got {len(entries)}."
            )
        for entry in entries:
            if entry.type not in allowed_operators:
                raise ValueError(
                    f"PDE '{pde_name}' boundary field '{field_name}' side "
                    f"'{side_name}' uses unsupported operator '{entry.type}'. "
                    f"Allowed operators: {sorted(allowed_operators)}."
                )
            if entry.type == "periodic":
                if entry.pair_with is None:
                    raise ValueError(
                        f"PDE '{pde_name}' boundary field '{field_name}' "
                        f"side '{side_name}' is periodic but missing 'pair_with'."
                    )
                try:
                    domain_geom.periodic_map(side_name, entry.pair_with)
                except KeyError as exc:
                    raise ValueError(
                        f"PDE '{pde_name}' boundary field '{field_name}' "
                        f"side '{side_name}' pairs with '{entry.pair_with}', but "
                        "the domain does not expose a periodic map for that pair."
                    ) from exc


def validate_scalar_standard_boundary_field(
    *,
    pde_name: str,
    field_name: str,
    boundary_field: BoundaryFieldConfig,
    domain_geom: DomainGeometry,
) -> None:
    """Validate the standard scalar one-operator-per-side model."""
    validate_boundary_field_structure(
        pde_name=pde_name,
        field_name=field_name,
        boundary_field=boundary_field,
        domain_geom=domain_geom,
        allowed_operators={"dirichlet", "neumann", "robin", "periodic"},
    )


def validate_vector_standard_boundary_field(
    *,
    pde_name: str,
    field_name: str,
    boundary_field: BoundaryFieldConfig,
    domain_geom: DomainGeometry,
    allowed_operators: set[str] | None = None,
) -> None:
    """Validate the standard vector one-operator-per-side model."""
    validate_boundary_field_structure(
        pde_name=pde_name,
        field_name=field_name,
        boundary_field=boundary_field,
        domain_geom=domain_geom,
        allowed_operators=(
            {"dirichlet", "neumann", "periodic"}
            if allowed_operators is None
            else allowed_operators
        ),
    )
