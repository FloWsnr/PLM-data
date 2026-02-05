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
        description: Human-readable description.
    """

    name: str
    description: str


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
        supported_dimensions: List of supported spatial dimensions (1, 2, 3).
            Must be explicitly set by each PDE preset.
    """

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    num_fields: int
    field_names: list[str]
    reference: str | None = None
    supported_dimensions: list[int] = field(default_factory=list)  # Must be explicitly set


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

    def validate_dimension(self, ndim: int) -> None:
        """Validate that the PDE supports the given number of dimensions.

        Args:
            ndim: Number of spatial dimensions (1, 2, or 3).

        Raises:
            ValueError: If the PDE does not support this dimensionality.
        """
        if not self.metadata.supported_dimensions:
            raise ValueError(
                f"PDE '{self.metadata.name}' has no supported_dimensions defined. "
                "Each PDE must explicitly specify supported_dimensions in its metadata."
            )
        if ndim not in self.metadata.supported_dimensions:
            supported_str = ", ".join(f"{d}D" for d in self.metadata.supported_dimensions)
            raise ValueError(
                f"PDE '{self.metadata.name}' does not support {ndim}D simulations. "
                f"Supported: {supported_str}"
            )

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

    def _infer_ndim_from_bc(self, bc: Any) -> int:
        """Infer dimensionality (1/2/3) from a BC object/dict.

        Most call-sites pass a BoundaryConfig, but some presets/tests may pass a
        raw dict. Inferring ndim here avoids accidentally dropping z-boundaries
        when running 3D simulations.
        """
        # BoundaryConfig
        if hasattr(bc, "x_minus"):
            # Prefer explicit z/y config if present
            z_minus = getattr(bc, "z_minus", None)
            z_plus = getattr(bc, "z_plus", None)
            if z_minus is not None or z_plus is not None:
                return 3
            y_minus = getattr(bc, "y_minus", None)
            y_plus = getattr(bc, "y_plus", None)
            if y_minus is not None or y_plus is not None:
                return 2
            return 1

        # Dict-style config: look for side keys
        if isinstance(bc, dict):
            if "z-" in bc or "z+" in bc:
                return 3
            if "y-" in bc or "y+" in bc:
                return 2
            return 1

        # Fallback: keep previous behavior
        return 2

    def _convert_bc(self, bc: Any, ndim: int | None = None) -> dict[str, Any]:
        """Convert boundary condition config to py-pde format.

        Args:
            bc: BoundaryConfig object or a raw dict.
            ndim: Number of spatial dimensions. If None, inferred from `bc`.

        Returns:
            BC specs in py-pde format.
        """
        if ndim is None:
            ndim = self._infer_ndim_from_bc(bc)
        return BoundaryConditionFactory.convert_config(bc, ndim)

    def _get_pde_bc_kwargs(self, bc: Any, ndim: int | None = None) -> dict:
        """Get bc kwargs for PDE constructor.

        Args:
            bc: Boundary condition configuration (BoundaryConfig)
            ndim: Number of spatial dimensions. If None, inferred from `bc`.

        Returns:
            Dictionary with {'bc': ...}
        """
        return {"bc": self._convert_bc(bc, ndim)}


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
            Example: {"u": {"type": "gaussian-blob", "params": {...}}}
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
