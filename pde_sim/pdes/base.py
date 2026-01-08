"""Abstract base classes for PDE presets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pde_sim.core.config import BoundaryConfig

from pde import CartesianGrid, FieldCollection, PDE, ScalarField

from pde_sim.boundaries import BoundaryConditionFactory
from pde_sim.initial_conditions import create_initial_condition


@dataclass
class PDEParameter:
    """Definition of a PDE parameter.

    Attributes:
        name: Parameter name (used in equations and config).
        default: Default value for the parameter.
        description: Human-readable description.
        min_value: Optional minimum allowed value.
        max_value: Optional maximum allowed value.
    """

    name: str
    default: float
    description: str
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class PDEMetadata:
    """Metadata describing a PDE preset.

    Attributes:
        name: Unique identifier for the preset.
        category: Category (basic, biology, physics, fluids).
        description: Human-readable description.
        equations: Dictionary mapping variable names to equation expressions.
        parameters: List of configurable parameters.
        num_fields: Number of fields in the system.
        field_names: Names of the fields.
        reference: Optional reference to literature or documentation.
    """

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    num_fields: int
    field_names: list[str]
    reference: str | None = None


class PDEPreset(ABC):
    """Abstract base class for all PDE presets.

    Subclasses must implement:
        - metadata property
        - create_pde method
        - create_initial_state method
    """

    @property
    @abstractmethod
    def metadata(self) -> PDEMetadata:
        """Return metadata describing this PDE."""
        pass

    @abstractmethod
    def create_pde(
        self,
        parameters: dict[str, float],
        bc: Any,
        grid: CartesianGrid,
    ) -> PDE:
        """Create the py-pde PDE instance.

        Args:
            parameters: Dictionary of parameter values.
            bc: Boundary condition specification (BoundaryConfig).
            grid: The computational grid.

        Returns:
            Configured PDE instance.
        """
        pass

    @abstractmethod
    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
    ) -> ScalarField | FieldCollection:
        """Create the initial field state.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.

        Returns:
            Initial field state.
        """
        pass

    def get_default_parameters(self) -> dict[str, float]:
        """Return default parameter values.

        Returns:
            Dictionary mapping parameter names to default values.
        """
        return {p.name: p.default for p in self.metadata.parameters}

    def validate_parameters(self, params: dict[str, float]) -> None:
        """Validate parameter values are within bounds.

        Args:
            params: Dictionary of parameter values to validate.

        Raises:
            ValueError: If a parameter is outside its allowed range.
        """
        for p in self.metadata.parameters:
            if p.name in params:
                val = params[p.name]
                if p.min_value is not None and val < p.min_value:
                    raise ValueError(f"{p.name} must be >= {p.min_value}, got {val}")
                if p.max_value is not None and val > p.max_value:
                    raise ValueError(f"{p.name} must be <= {p.max_value}, got {val}")

    def get_equations_for_metadata(
        self, parameters: dict[str, float]
    ) -> dict[str, str]:
        """Get equations with parameter values substituted.

        Args:
            parameters: Dictionary of parameter values.

        Returns:
            Dictionary of equation expressions with values substituted.
        """
        # Default implementation just returns the template equations
        # Subclasses can override to provide filled-in versions
        return self.metadata.equations.copy()

    def _convert_bc(self, bc: Any) -> dict[str, Any]:
        """Convert boundary condition config to py-pde format.

        Args:
            bc: BoundaryConfig object.

        Returns:
            BC specs in py-pde format with x-, x+, y-, y+ keys.
        """
        from pde_sim.core.config import BoundaryConfig

        if isinstance(bc, BoundaryConfig):
            return BoundaryConditionFactory.convert_config(bc)
        # Handle dict (should have x-, x+, y-, y+ keys)
        return BoundaryConditionFactory.convert_config(bc)

    def _get_pde_bc_kwargs(self, bc: Any) -> dict:
        """Get bc kwargs for PDE constructor.

        Args:
            bc: Boundary condition configuration (BoundaryConfig)

        Returns:
            Dictionary with {'bc': ...}
        """
        return {"bc": self._convert_bc(bc)}


class ScalarPDEPreset(PDEPreset):
    """Base class for single-field scalar PDEs.

    Provides a default implementation of create_initial_state
    that generates a single ScalarField.
    """

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> ScalarField:
        """Create scalar initial field.

        Args:
            grid: The computational grid.
            ic_type: Type of initial condition.
            ic_params: Parameters for the initial condition.
            **kwargs: Additional arguments (ignored for scalar PDEs).

        Returns:
            Initial scalar field.
        """
        return create_initial_condition(grid, ic_type, ic_params)


@dataclass
class FieldConfig:
    """Configuration for a single field's initial condition."""

    ic_type: str
    ic_params: dict[str, Any] = field(default_factory=dict)


class MultiFieldPDEPreset(PDEPreset):
    """Base class for multi-field/coupled PDEs.

    Provides a default implementation of create_initial_state
    that generates a FieldCollection with one field per field_name.
    """

    def create_initial_state(
        self,
        grid: CartesianGrid,
        ic_type: str,
        ic_params: dict[str, Any],
        **kwargs,
    ) -> FieldCollection:
        """Create multi-field initial state.

        Args:
            grid: The computational grid.
            ic_type: Default initial condition type.
            ic_params: Parameters which may include per-field overrides.
            **kwargs: Additional arguments (parameters, bc) for specific PDEs.

        Returns:
            Initial field collection.

        Notes:
            ic_params can specify per-field ICs using field names as keys.
            Example: {"u": {"type": "gaussian-blobs", "params": {...}}}
        """
        fields = []
        field_names = self.metadata.field_names

        for name in field_names:
            if name in ic_params and isinstance(ic_params[name], dict):
                # Per-field IC specification
                field_ic = ic_params[name]
                field_type = field_ic.get("type", ic_type)
                field_params = field_ic.get("params", {})
                ic = create_initial_condition(grid, field_type, field_params)
            else:
                # Use global IC type with global params
                ic = create_initial_condition(grid, ic_type, ic_params)

            # Set field label
            ic.label = name
            fields.append(ic)

        return FieldCollection(fields)
