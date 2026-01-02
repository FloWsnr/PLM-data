"""Boundary condition factory for py-pde."""

from typing import Any

from pde import CartesianGrid


class BoundaryConditionFactory:
    """Factory for creating py-pde boundary condition specifications.

    Converts human-readable BC names to py-pde format.
    """

    # Mapping from config names to py-pde BC specs
    BC_MAP = {
        # Periodic
        "periodic": "periodic",
        # Neumann (no-flux) - use derivative:0 for zero normal derivative
        "neumann": {"derivative": 0},
        "no-flux": {"derivative": 0},
        "zero-flux": {"derivative": 0},
        # Dirichlet (fixed value)
        "dirichlet": {"value": 0},
        "zero": {"value": 0},
        "wall": {"value": 0},
        "fixed": {"value": 0},
        # Derivative conditions
        "zero-derivative": {"derivative": 0},
    }

    @classmethod
    def convert(cls, bc_type: str, value: float | None = None) -> Any:
        """Convert a BC type name to py-pde format.

        Args:
            bc_type: The boundary condition type name.
            value: Optional value for Dirichlet conditions.

        Returns:
            py-pde compatible boundary condition specification.

        Raises:
            ValueError: If the BC type is not recognized.
        """
        bc_lower = bc_type.lower()

        if bc_lower in cls.BC_MAP:
            bc_spec = cls.BC_MAP[bc_lower]
            # If a specific value is provided for Dirichlet, use it
            if value is not None and isinstance(bc_spec, dict) and "value" in bc_spec:
                return {"value": value}
            return bc_spec

        # Check for parameterized BCs like "dirichlet:0.5"
        if ":" in bc_type:
            parts = bc_type.split(":")
            base_type = parts[0].lower()
            param_value = float(parts[1])

            if base_type in ("dirichlet", "value"):
                return {"value": param_value}
            elif base_type in ("neumann", "derivative"):
                return {"derivative": param_value}

        raise ValueError(
            f"Unknown boundary condition type: {bc_type}. "
            f"Available: {list(cls.BC_MAP.keys())}"
        )

    @classmethod
    def convert_config(cls, bc_config: dict[str, str]) -> list:
        """Convert a BC config dict to py-pde format.

        Args:
            bc_config: Dictionary with 'x' and 'y' keys.

        Returns:
            List of BC specs for [x, y] axes.
        """
        x_bc = cls.convert(bc_config.get("x", "periodic"))
        y_bc = cls.convert(bc_config.get("y", "periodic"))
        return [x_bc, y_bc]

    @classmethod
    def get_periodic_flags(cls, bc_config: dict[str, str]) -> list[bool]:
        """Get periodic flags for each axis.

        Args:
            bc_config: Dictionary with 'x' and 'y' keys.

        Returns:
            List of boolean flags [x_periodic, y_periodic].
        """
        x_periodic = bc_config.get("x", "periodic").lower() == "periodic"
        y_periodic = bc_config.get("y", "periodic").lower() == "periodic"
        return [x_periodic, y_periodic]


def create_grid_with_bc(
    resolution: int,
    domain_size: float,
    bc_config: dict[str, str],
) -> CartesianGrid:
    """Create a CartesianGrid with appropriate periodicity.

    Args:
        resolution: Number of grid points in each dimension.
        domain_size: Physical size of the domain.
        bc_config: Dictionary with 'x' and 'y' boundary conditions.

    Returns:
        Configured CartesianGrid.
    """
    periodic = BoundaryConditionFactory.get_periodic_flags(bc_config)

    return CartesianGrid(
        bounds=[[0, domain_size], [0, domain_size]],
        shape=[resolution, resolution],
        periodic=periodic,
    )
