"""Boundary-condition operator and family specifications."""

from dataclasses import dataclass, field

_VALID_FIELD_SHAPES = {"scalar", "vector"}
_VALID_VALUE_SHAPES = {"field", "scalar", "vector"}


@dataclass(frozen=True)
class BoundaryOperatorParameterSpec:
    """Sampling-facing declaration for one boundary-operator parameter."""

    name: str
    kind: str
    description: str = ""
    hard_min: float | int | None = None
    hard_max: float | int | None = None
    sampling_min: float | int | None = None
    sampling_max: float | int | None = None


@dataclass(frozen=True)
class BoundaryOperatorSpec:
    """Config-facing declaration for one named boundary operator."""

    name: str
    value_shape: str | None
    requires_pair_with: bool = False
    operator_parameter_names: tuple[str, ...] = ()
    description: str = ""
    allowed_field_shapes: tuple[str, ...] = ("scalar", "vector")
    parameter_specs: dict[str, BoundaryOperatorParameterSpec] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.value_shape is not None and self.value_shape not in _VALID_VALUE_SHAPES:
            raise ValueError(
                f"Boundary operator '{self.name}' has unsupported value shape "
                f"'{self.value_shape}'."
            )
        invalid_shapes = set(self.allowed_field_shapes) - _VALID_FIELD_SHAPES
        if invalid_shapes:
            raise ValueError(
                f"Boundary operator '{self.name}' has unsupported field shapes "
                f"{sorted(invalid_shapes)}."
            )
        if not self.allowed_field_shapes:
            raise ValueError(
                f"Boundary operator '{self.name}' must allow at least one field shape."
            )

        parameter_names = tuple(self.parameter_specs)
        if self.operator_parameter_names and parameter_names:
            if set(self.operator_parameter_names) != set(parameter_names):
                raise ValueError(
                    f"Boundary operator '{self.name}' parameter names "
                    f"{sorted(self.operator_parameter_names)} do not match "
                    f"parameter specs {sorted(parameter_names)}."
                )
        elif parameter_names:
            object.__setattr__(self, "operator_parameter_names", parameter_names)
        for name, parameter_spec in self.parameter_specs.items():
            if name != parameter_spec.name:
                raise ValueError(
                    f"Boundary operator '{self.name}' parameter key '{name}' does "
                    f"not match BoundaryOperatorParameterSpec.name "
                    f"'{parameter_spec.name}'."
                )


@dataclass(frozen=True)
class BoundaryFamilySpec:
    """Reusable boundary-condition recipe available to domains and presets."""

    name: str
    description: str
    operators: tuple[str, ...]
    required_domain_roles: tuple[str, ...] = ()
    required_any_domain_roles: tuple[tuple[str, ...], ...] = ()
    supported_field_shapes: tuple[str, ...] = ("scalar", "vector")
    supported_dimensions: tuple[int, ...] = (1, 2, 3)
    requires_periodic_pairs: bool = False

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError(
                f"Boundary family '{self.name}' must allow at least one operator."
            )
        invalid_shapes = set(self.supported_field_shapes) - _VALID_FIELD_SHAPES
        if invalid_shapes:
            raise ValueError(
                f"Boundary family '{self.name}' has unsupported field shapes "
                f"{sorted(invalid_shapes)}."
            )
        if not self.supported_field_shapes:
            raise ValueError(
                f"Boundary family '{self.name}' must allow at least one field shape."
            )
        if not self.supported_dimensions:
            raise ValueError(
                f"Boundary family '{self.name}' must support at least one dimension."
            )
        for dimension in self.supported_dimensions:
            if dimension not in {1, 2, 3}:
                raise ValueError(
                    f"Boundary family '{self.name}' has unsupported dimension "
                    f"{dimension}."
                )
        for alternatives in self.required_any_domain_roles:
            if not alternatives:
                raise ValueError(
                    f"Boundary family '{self.name}' has an empty role-alternative "
                    "constraint."
                )

    def is_compatible_with_domain(self, domain_spec) -> bool:
        """Return whether a domain spec exposes the roles this family needs."""
        if domain_spec.dimension not in self.supported_dimensions:
            return False
        domain_roles = set(domain_spec.boundary_roles)
        if not set(self.required_domain_roles).issubset(domain_roles):
            return False
        for alternatives in self.required_any_domain_roles:
            if not domain_roles.intersection(alternatives):
                return False
        return not self.requires_periodic_pairs or bool(domain_spec.periodic_pairs)


_BOUNDARY_OPERATOR_REGISTRY: dict[str, BoundaryOperatorSpec] = {}
_BOUNDARY_FAMILY_REGISTRY: dict[str, BoundaryFamilySpec] = {}

COMMON_BOUNDARY_FAMILIES = (
    "all_dirichlet",
    "all_neumann",
    "all_robin",
)


def register_boundary_operator_spec(spec: BoundaryOperatorSpec) -> BoundaryOperatorSpec:
    """Register metadata for one boundary operator."""
    _BOUNDARY_OPERATOR_REGISTRY[spec.name] = spec
    return spec


def register_boundary_family_spec(spec: BoundaryFamilySpec) -> BoundaryFamilySpec:
    """Register metadata for one boundary-condition family."""
    unknown_operators = set(spec.operators) - set(_BOUNDARY_OPERATOR_REGISTRY)
    if unknown_operators:
        raise ValueError(
            f"Boundary-condition family '{spec.name}' references unknown operators "
            f"{sorted(unknown_operators)}."
        )
    _BOUNDARY_FAMILY_REGISTRY[spec.name] = spec
    return spec


def list_boundary_operator_specs() -> dict[str, BoundaryOperatorSpec]:
    """Return all registered boundary-operator specs."""
    return dict(_BOUNDARY_OPERATOR_REGISTRY)


def list_boundary_family_specs() -> dict[str, BoundaryFamilySpec]:
    """Return all registered boundary-condition family specs."""
    return dict(_BOUNDARY_FAMILY_REGISTRY)


def get_boundary_operator_spec(name: str) -> BoundaryOperatorSpec:
    """Return one registered boundary-operator spec."""
    if name not in _BOUNDARY_OPERATOR_REGISTRY:
        available = ", ".join(sorted(_BOUNDARY_OPERATOR_REGISTRY))
        raise ValueError(f"Unknown boundary operator '{name}'. Available: {available}")
    return _BOUNDARY_OPERATOR_REGISTRY[name]


def get_boundary_family_spec(name: str) -> BoundaryFamilySpec:
    """Return one registered boundary-condition family spec."""
    if name not in _BOUNDARY_FAMILY_REGISTRY:
        available = ", ".join(sorted(_BOUNDARY_FAMILY_REGISTRY))
        raise ValueError(
            f"Unknown boundary-condition family '{name}'. Available: {available}"
        )
    return _BOUNDARY_FAMILY_REGISTRY[name]


def has_boundary_family_spec(name: str) -> bool:
    """Return whether a boundary-condition family has been registered."""
    return name in _BOUNDARY_FAMILY_REGISTRY
