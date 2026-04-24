"""Initial-condition operator specifications."""

from dataclasses import dataclass

_VALID_FIELD_SHAPES = {"scalar", "vector"}
_VALID_DIMENSIONS = {1, 2, 3}


@dataclass(frozen=True)
class InitialConditionOperatorParameterSpec:
    """Sampling-facing declaration for one initial-condition operator parameter."""

    name: str
    kind: str
    description: str = ""
    required: bool = True
    length: int | None = None
    hard_min: float | int | None = None
    hard_max: float | int | None = None
    sampling_min: float | int | None = None
    sampling_max: float | int | None = None


@dataclass(frozen=True)
class InitialConditionOperatorSpec:
    """Reusable initial-condition operator available to domains and PDEs."""

    name: str
    description: str
    parameters: dict[str, InitialConditionOperatorParameterSpec]
    supported_dimensions: tuple[int, ...] = (1, 2, 3)
    supported_field_shapes: tuple[str, ...] = ("scalar", "vector")
    supports_vector_field_level: bool = False
    requires_seed: bool = False
    common_scalar_operator: bool = False

    def __post_init__(self) -> None:
        if not self.supported_dimensions:
            raise ValueError(
                f"Initial-condition operator '{self.name}' must support at least "
                "one dimension."
            )
        invalid_dimensions = set(self.supported_dimensions) - _VALID_DIMENSIONS
        if invalid_dimensions:
            raise ValueError(
                f"Initial-condition operator '{self.name}' has unsupported "
                f"dimensions {sorted(invalid_dimensions)}."
            )

        invalid_shapes = set(self.supported_field_shapes) - _VALID_FIELD_SHAPES
        if invalid_shapes:
            raise ValueError(
                f"Initial-condition operator '{self.name}' has unsupported field "
                f"shapes {sorted(invalid_shapes)}."
            )
        if not self.supported_field_shapes:
            raise ValueError(
                f"Initial-condition operator '{self.name}' must support at least "
                "one field shape."
            )

        for name, parameter in self.parameters.items():
            if name != parameter.name:
                raise ValueError(
                    f"Initial-condition operator '{self.name}' parameter key "
                    f"'{name}' does not match InitialConditionOperatorParameterSpec.name "
                    f"'{parameter.name}'."
                )

    def is_compatible_with_domain(self, domain_spec) -> bool:
        """Return whether a domain spec has a compatible dimension."""
        return domain_spec.dimension in self.supported_dimensions


_INITIAL_CONDITION_OPERATOR_REGISTRY: dict[str, InitialConditionOperatorSpec] = {}

COMMON_SCALAR_INITIAL_CONDITION_OPERATORS = (
    "constant",
    "gaussian_bump",
    "gaussian_blobs",
    "gaussian_noise",
    "gaussian_wave_packet",
    "quadrants",
    "radial_cosine",
    "sine_waves",
    "step",
)


def register_initial_condition_operator_spec(
    spec: InitialConditionOperatorSpec,
) -> InitialConditionOperatorSpec:
    """Register metadata for one initial-condition operator."""
    _INITIAL_CONDITION_OPERATOR_REGISTRY[spec.name] = spec
    return spec


def list_initial_condition_operator_specs() -> dict[str, InitialConditionOperatorSpec]:
    """Return all registered initial-condition operator specs."""
    return dict(_INITIAL_CONDITION_OPERATOR_REGISTRY)


def get_initial_condition_operator_spec(name: str) -> InitialConditionOperatorSpec:
    """Return one registered initial-condition operator spec."""
    if name not in _INITIAL_CONDITION_OPERATOR_REGISTRY:
        available = ", ".join(sorted(_INITIAL_CONDITION_OPERATOR_REGISTRY))
        raise ValueError(
            f"Unknown initial-condition operator '{name}'. Available: {available}"
        )
    return _INITIAL_CONDITION_OPERATOR_REGISTRY[name]


def has_initial_condition_operator_spec(name: str) -> bool:
    """Return whether an initial-condition operator has been registered."""
    return name in _INITIAL_CONDITION_OPERATOR_REGISTRY
