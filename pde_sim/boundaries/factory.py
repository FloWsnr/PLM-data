"""Boundary condition factory for py-pde."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pde import CartesianGrid

if TYPE_CHECKING:
    from pde_sim.core.config import BoundaryConfig


class BoundaryConditionFactory:
    """Factory for creating py-pde boundary condition specifications.

    Supports three BC types:
    - `periodic` - periodic boundary (no value needed)
    - `neumann:VALUE` - fixed derivative (value required, e.g., neumann:0)
    - `dirichlet:VALUE` - fixed value (value required, e.g., dirichlet:0)
    """

    @classmethod
    def convert(cls, bc_str: str) -> Any:
        """Convert BC string to py-pde format.

        Args:
            bc_str: One of: "periodic", "neumann:VALUE", "dirichlet:VALUE"

        Returns:
            py-pde compatible boundary condition specification.

        Raises:
            ValueError: If the BC type is not recognized or value is missing.
        """
        bc_lower = bc_str.lower().strip()

        if bc_lower == "periodic":
            return "periodic"

        if ":" not in bc_str:
            raise ValueError(
                f"BC '{bc_str}' requires a value. Use 'neumann:0' or 'dirichlet:0'"
            )

        bc_type, value_str = bc_str.split(":", 1)
        value = float(value_str)

        if bc_type.lower() == "neumann":
            return {"derivative": value}
        elif bc_type.lower() == "dirichlet":
            return {"value": value}
        else:
            raise ValueError(
                f"Unknown BC type: {bc_type}. Use periodic, neumann, or dirichlet"
            )

    @classmethod
    def convert_config(cls, bc_config: BoundaryConfig | dict[str, str]) -> dict[str, Any]:
        """Convert BC config to py-pde format.

        Args:
            bc_config: BoundaryConfig object or dict with x-, x+, y-, y+ keys.

        Returns:
            Dictionary with side-keyed BC specs for py-pde.
            Format: {"x-": bc, "x+": bc, "y-": bc, "y+": bc}
        """
        if hasattr(bc_config, "x_minus"):
            # BoundaryConfig object
            return {
                "x-": cls.convert(bc_config.x_minus),
                "x+": cls.convert(bc_config.x_plus),
                "y-": cls.convert(bc_config.y_minus),
                "y+": cls.convert(bc_config.y_plus),
            }
        else:
            # Dict with x-, x+, y-, y+ keys
            return {
                "x-": cls.convert(bc_config.get("x-", "periodic")),
                "x+": cls.convert(bc_config.get("x+", "periodic")),
                "y-": cls.convert(bc_config.get("y-", "periodic")),
                "y+": cls.convert(bc_config.get("y+", "periodic")),
            }

    @classmethod
    def convert_field_bc(cls, field_bc: dict[str, str]) -> dict[str, Any]:
        """Convert field-specific BC dict to py-pde format.

        Args:
            field_bc: Dict with x-, x+, y-, y+ keys (already merged with defaults).

        Returns:
            BC specification in py-pde format.
        """
        return {
            "x-": cls.convert(field_bc["x-"]),
            "x+": cls.convert(field_bc["x+"]),
            "y-": cls.convert(field_bc["y-"]),
            "y+": cls.convert(field_bc["y+"]),
        }

    @classmethod
    def get_periodic_flags(cls, bc_config: BoundaryConfig) -> list[bool]:
        """Get periodic flags for each axis.

        Args:
            bc_config: Boundary configuration.

        Returns:
            List of boolean flags [x_periodic, y_periodic].
        """
        x_periodic = bc_config.x_minus == "periodic" and bc_config.x_plus == "periodic"
        y_periodic = bc_config.y_minus == "periodic" and bc_config.y_plus == "periodic"
        return [x_periodic, y_periodic]


def create_grid_with_bc(
    resolution: int,
    domain_size: float,
    bc_config: BoundaryConfig,
) -> CartesianGrid:
    """Create a CartesianGrid with appropriate periodicity.

    Args:
        resolution: Number of grid points in each dimension.
        domain_size: Physical size of the domain.
        bc_config: Boundary configuration.

    Returns:
        Configured CartesianGrid.
    """
    periodic = BoundaryConditionFactory.get_periodic_flags(bc_config)

    return CartesianGrid(
        bounds=[[0, domain_size], [0, domain_size]],
        shape=[resolution, resolution],
        periodic=periodic,
    )
