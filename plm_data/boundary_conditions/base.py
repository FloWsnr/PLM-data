"""Boundary-condition operator specifications."""

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


_BOUNDARY_OPERATOR_REGISTRY: dict[str, BoundaryOperatorSpec] = {}


def register_boundary_operator_spec(spec: BoundaryOperatorSpec) -> BoundaryOperatorSpec:
    """Register metadata for one boundary operator."""
    _BOUNDARY_OPERATOR_REGISTRY[spec.name] = spec
    return spec


def list_boundary_operator_specs() -> dict[str, BoundaryOperatorSpec]:
    """Return all registered boundary-operator specs."""
    return dict(_BOUNDARY_OPERATOR_REGISTRY)


def get_boundary_operator_spec(name: str) -> BoundaryOperatorSpec:
    """Return one registered boundary-operator spec."""
    if name not in _BOUNDARY_OPERATOR_REGISTRY:
        available = ", ".join(sorted(_BOUNDARY_OPERATOR_REGISTRY))
        raise ValueError(f"Unknown boundary operator '{name}'. Available: {available}")
    return _BOUNDARY_OPERATOR_REGISTRY[name]
