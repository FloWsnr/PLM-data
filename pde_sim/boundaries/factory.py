"""Boundary condition factory for py-pde."""

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
    def convert(
        cls, bc_str: str, parameters: dict[str, float] | None = None
    ) -> Any:
        """Convert BC string to py-pde format.

        Args:
            bc_str: One of: "periodic", "neumann:VALUE", "dirichlet:VALUE".
                VALUE can be a number (e.g., ``dirichlet:0.5``) or a parameter
                name (e.g., ``dirichlet:U``) that will be looked up in
                *parameters*.
            parameters: PDE parameter dict used to resolve named references.

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
        value = cls._resolve_value(value_str, parameters)

        if bc_type.lower() == "neumann":
            return {"derivative": value}
        elif bc_type.lower() == "dirichlet":
            return {"value": value}
        else:
            raise ValueError(
                f"Unknown BC type: {bc_type}. Use periodic, neumann, or dirichlet"
            )

    @classmethod
    def _resolve_value(
        cls, value_str: str, parameters: dict[str, float] | None
    ) -> float:
        """Resolve a BC value string to a float.

        Tries numeric parsing first. If that fails, looks up *value_str* as a
        parameter name in *parameters*.

        Args:
            value_str: Numeric literal or parameter name.
            parameters: PDE parameter dict.

        Returns:
            Resolved float value.

        Raises:
            ValueError: If the value cannot be resolved.
        """
        try:
            return float(value_str)
        except ValueError:
            pass

        if parameters is not None and value_str in parameters:
            return float(parameters[value_str])

        if parameters is None:
            raise ValueError(
                f"BC value '{value_str}' is not numeric and no parameters dict "
                "was provided to resolve it"
            )
        raise ValueError(
            f"BC value '{value_str}' is not numeric and not found in parameters. "
            f"Available parameters: {sorted(parameters.keys())}"
        )

    @classmethod
    def convert_config(
        cls,
        bc_config: "BoundaryConfig | dict[str, str]",
        ndim: int = 2,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Convert BC config to py-pde format.

        Args:
            bc_config: BoundaryConfig object or dict with boundary keys.
            ndim: Number of spatial dimensions.
            parameters: PDE parameter dict for resolving named BC values.

        Returns:
            Dictionary with side-keyed BC specs for py-pde.
            1D: {"x-": bc, "x+": bc}
            2D: {"x-": bc, "x+": bc, "y-": bc, "y+": bc}
            3D: {"x-": bc, "x+": bc, "y-": bc, "y+": bc, "z-": bc, "z+": bc}
        """
        if hasattr(bc_config, "x_minus"):
            # BoundaryConfig object
            result = {
                "x-": cls.convert(bc_config.x_minus, parameters),
                "x+": cls.convert(bc_config.x_plus, parameters),
            }
            if ndim >= 2 and bc_config.y_minus is not None:
                result["y-"] = cls.convert(bc_config.y_minus, parameters)
                result["y+"] = cls.convert(bc_config.y_plus, parameters)
            if ndim >= 3 and bc_config.z_minus is not None:
                result["z-"] = cls.convert(bc_config.z_minus, parameters)
                result["z+"] = cls.convert(bc_config.z_plus, parameters)
            return result
        else:
            # Dict with boundary keys
            result = {
                "x-": cls.convert(bc_config.get("x-", "periodic"), parameters),
                "x+": cls.convert(bc_config.get("x+", "periodic"), parameters),
            }
            if ndim >= 2:
                result["y-"] = cls.convert(bc_config.get("y-", "periodic"), parameters)
                result["y+"] = cls.convert(bc_config.get("y+", "periodic"), parameters)
            if ndim >= 3:
                result["z-"] = cls.convert(bc_config.get("z-", "periodic"), parameters)
                result["z+"] = cls.convert(bc_config.get("z+", "periodic"), parameters)
            return result

    @classmethod
    def convert_field_bc(
        cls,
        field_bc: dict[str, str],
        ndim: int = 2,
        parameters: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Convert field-specific BC dict to py-pde format.

        Args:
            field_bc: Dict with boundary keys (already merged with defaults).
            ndim: Number of spatial dimensions.
            parameters: PDE parameter dict for resolving named BC values.

        Returns:
            BC specification in py-pde format.
        """
        result = {
            "x-": cls.convert(field_bc["x-"], parameters),
            "x+": cls.convert(field_bc["x+"], parameters),
        }
        if ndim >= 2 and "y-" in field_bc:
            result["y-"] = cls.convert(field_bc["y-"], parameters)
            result["y+"] = cls.convert(field_bc["y+"], parameters)
        if ndim >= 3 and "z-" in field_bc:
            result["z-"] = cls.convert(field_bc["z-"], parameters)
            result["z+"] = cls.convert(field_bc["z+"], parameters)
        return result

    @classmethod
    def get_periodic_flags(cls, bc_config: "BoundaryConfig", ndim: int = 2) -> list[bool]:
        """Get periodic flags for each axis.

        Args:
            bc_config: Boundary configuration.
            ndim: Number of spatial dimensions (1, 2, or 3).

        Returns:
            List of boolean flags for each axis.
            1D: [x_periodic]
            2D: [x_periodic, y_periodic]
            3D: [x_periodic, y_periodic, z_periodic]
        """
        x_periodic = bc_config.x_minus == "periodic" and bc_config.x_plus == "periodic"
        flags = [x_periodic]

        if ndim >= 2:
            y_periodic = (
                bc_config.y_minus is not None
                and bc_config.y_minus == "periodic"
                and bc_config.y_plus == "periodic"
            )
            flags.append(y_periodic)

        if ndim >= 3:
            z_periodic = (
                bc_config.z_minus is not None
                and bc_config.z_minus == "periodic"
                and bc_config.z_plus == "periodic"
            )
            flags.append(z_periodic)

        return flags


def create_grid_with_bc(
    resolution: list[int],
    domain_size: list[float],
    bc_config: "BoundaryConfig",
) -> CartesianGrid:
    """Create a CartesianGrid with appropriate periodicity.

    Args:
        resolution: Number of grid points for each dimension [nx], [nx, ny], or [nx, ny, nz].
        domain_size: Physical size for each dimension, matching resolution length.
        bc_config: Boundary configuration.

    Returns:
        Configured CartesianGrid.
    """
    ndim = len(resolution)
    if len(domain_size) != ndim:
        raise ValueError(
            f"resolution has {ndim} dimensions but domain_size has {len(domain_size)}"
        )

    # Validate boundary conditions for this dimension
    bc_config.validate_for_ndim(ndim)

    # Get periodic flags for each axis
    periodic = BoundaryConditionFactory.get_periodic_flags(bc_config, ndim)

    # Build bounds: [[0, Lx]] for 1D, [[0, Lx], [0, Ly]] for 2D, etc.
    bounds = [[0.0, domain_size[i]] for i in range(ndim)]

    return CartesianGrid(
        bounds=bounds,
        shape=resolution,
        periodic=periodic,
    )
