"""PDE specifications and public extension contracts."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plm_data.sampling.context import SamplingContext

from plm_data.boundary_conditions import (
    BoundaryOperatorSpec,
    MAXWELL_BOUNDARY_OPERATORS,
    SCALAR_STANDARD_BOUNDARY_OPERATORS,
    VECTOR_STANDARD_BOUNDARY_OPERATORS,
)

_COMPONENT_LABELS = ("x", "y", "z")
_VALID_STOCHASTIC_COUPLINGS = {
    "additive",
    "multiplicative_self",
    "saturating_self",
}
GENERIC_STOCHASTIC_COUPLINGS = ("additive", "multiplicative_self")
SATURATING_STOCHASTIC_COUPLINGS = (
    "additive",
    "multiplicative_self",
    "saturating_self",
)
__all__ = [
    "BoundaryFieldSpec",
    "BoundaryOperatorSpec",
    "CoefficientSpec",
    "ConcreteOutputSpec",
    "GENERIC_STOCHASTIC_COUPLINGS",
    "InputSpec",
    "MAXWELL_BOUNDARY_OPERATORS",
    "OutputSpec",
    "PDEParameter",
    "PDEParameterSampler",
    "PDESpec",
    "SATURATING_STOCHASTIC_COUPLINGS",
    "SCALAR_STANDARD_BOUNDARY_OPERATORS",
    "StateSpec",
    "VECTOR_STANDARD_BOUNDARY_OPERATORS",
]

PDEParameterSampler = Callable[["SamplingContext", str], float | int]


def _component_labels(gdim: int) -> tuple[str, ...]:
    return _COMPONENT_LABELS[:gdim]


@dataclass(frozen=True)
class PDEParameter:
    """A configurable PDE parameter."""

    name: str
    description: str
    kind: str = "float"
    hard_min: float | int | None = None
    hard_max: float | int | None = None
    sampling_min: float | int | None = None
    sampling_max: float | int | None = None
    default: float | int | None = None
    sampler: PDEParameterSampler | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.kind not in {"float", "int"}:
            raise ValueError(
                f"PDE parameter '{self.name}' has unsupported kind '{self.kind}'."
            )
        if (self.sampling_min is None) != (self.sampling_max is None):
            raise ValueError(
                f"PDE parameter '{self.name}' must define both sampling_min and "
                "sampling_max, or neither."
            )
        if (
            self.sampling_min is not None
            and self.sampling_max is not None
            and self.sampling_min > self.sampling_max
        ):
            raise ValueError(
                f"PDE parameter '{self.name}' has sampling_min greater than "
                "sampling_max."
            )
        if (
            self.hard_min is not None
            and self.hard_max is not None
            and self.hard_min > self.hard_max
        ):
            raise ValueError(
                f"PDE parameter '{self.name}' has hard_min greater than hard_max."
            )
        if self.sampling_min is not None:
            self.validate_value(
                self.sampling_min,
                context=f"PDE parameter '{self.name}' sampling_min",
            )
            self.validate_value(
                self.sampling_max,
                context=f"PDE parameter '{self.name}' sampling_max",
            )
        if self.default is not None:
            self.validate_value(
                self.default,
                context=f"PDE parameter '{self.name}' default",
            )

    @property
    def has_sampling_rule(self) -> bool:
        """Return whether this parameter can be sampled directly from its spec."""
        return (
            self.default is not None
            or self.sampler is not None
            or (self.sampling_min is not None and self.sampling_max is not None)
        )

    def validate_value(self, value: float | int | None, *, context: str) -> None:
        """Validate one concrete parameter value against declared hard bounds."""
        if value is None:
            raise ValueError(f"{context} cannot be None.")
        if self.kind == "int":
            rounded = int(round(value))
            if abs(float(value) - rounded) > 1.0e-9:
                raise ValueError(f"{context} must be integer-valued. Got {value!r}.")
        if self.hard_min is not None and value < self.hard_min:
            raise ValueError(f"{context} must be >= {self.hard_min}. Got {value!r}.")
        if self.hard_max is not None and value > self.hard_max:
            raise ValueError(f"{context} must be <= {self.hard_max}. Got {value!r}.")


@dataclass(frozen=True)
class CoefficientSpec:
    """A config-facing coefficient field used in variational forms."""

    name: str
    shape: str
    description: str = ""
    allow_randomization: bool = False

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)


@dataclass(frozen=True)
class InputSpec:
    """Config-facing input declaration for a PDE."""

    name: str
    shape: str
    allow_source: bool
    allow_initial_condition: bool

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)


@dataclass(frozen=True)
class StateSpec:
    """A solved state variable of a PDE."""

    name: str
    shape: str
    stochastic_couplings: tuple[str, ...] = ()

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)


@dataclass(frozen=True)
class BoundaryFieldSpec:
    """Config-facing declaration for one BC-addressable field."""

    name: str
    shape: str
    operators: dict[str, BoundaryOperatorSpec]
    description: str = ""

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)


@dataclass(frozen=True)
class OutputSpec:
    """A config-selectable output exported by a PDE."""

    name: str
    shape: str
    output_mode: str
    source_name: str
    source_kind: str = "state"

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)

    def concrete_names(self, gdim: int, mode: str) -> list[str]:
        if mode == "hidden":
            return []
        if self.shape == "scalar":
            if mode != "scalar":
                raise ValueError(
                    f"Scalar output '{self.name}' does not support mode '{mode}'"
                )
            return [self.name]
        if mode != "components":
            raise ValueError(
                f"Vector output '{self.name}' does not support mode '{mode}'"
            )
        return [f"{self.name}_{label}" for label in self.component_labels(gdim)]

    def validate_output_mode(self, mode: str) -> None:
        allowed = {self.output_mode, "hidden"}
        if mode not in allowed:
            raise ValueError(
                f"Output '{self.name}' mode must be one of {sorted(allowed)}. "
                f"Got '{mode}'."
            )


@dataclass(frozen=True)
class ConcreteOutputSpec:
    """A concrete output array exported by a PDE."""

    name: str
    output_name: str
    source_name: str
    component: str | None = None


@dataclass(frozen=True)
class PDESpec:
    """Metadata and validation contract describing a PDE."""

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    inputs: dict[str, InputSpec]
    boundary_fields: dict[str, BoundaryFieldSpec]
    states: dict[str, StateSpec]
    outputs: dict[str, OutputSpec]
    static_fields: list[str]
    supported_dimensions: list[int]
    coefficients: dict[str, CoefficientSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen_parameters: set[str] = set()
        for parameter in self.parameters:
            if parameter.name in seen_parameters:
                raise ValueError(
                    f"PDE '{self.name}' declares duplicate parameter "
                    f"'{parameter.name}'."
                )
            seen_parameters.add(parameter.name)

        for name, spec in self.coefficients.items():
            if name != spec.name:
                raise ValueError(
                    f"PDE '{self.name}' coefficient key '{name}' does not match "
                    f"CoefficientSpec.name '{spec.name}'"
                )
            if spec.shape not in {"scalar", "vector"}:
                raise ValueError(
                    f"PDE '{self.name}' coefficient '{name}' has unsupported "
                    f"shape '{spec.shape}'"
                )
            if spec.allow_randomization and spec.shape != "scalar":
                raise ValueError(
                    f"PDE '{self.name}' coefficient '{name}' enables stochastic "
                    "randomization, but v1 only supports scalar coefficients."
                )

        for name, spec in self.inputs.items():
            if name != spec.name:
                raise ValueError(
                    f"PDE '{self.name}' input key '{name}' does not match "
                    f"InputSpec.name '{spec.name}'"
                )

        for name, spec in self.boundary_fields.items():
            if name != spec.name:
                raise ValueError(
                    f"PDE '{self.name}' boundary field key '{name}' does not "
                    f"match BoundaryFieldSpec.name '{spec.name}'"
                )
            if spec.shape not in {"scalar", "vector"}:
                raise ValueError(
                    f"PDE '{self.name}' boundary field '{name}' has unsupported "
                    f"shape '{spec.shape}'"
                )
            for operator_name, operator_spec in spec.operators.items():
                if operator_name != operator_spec.name:
                    raise ValueError(
                        f"PDE '{self.name}' boundary field '{name}' operator key "
                        f"'{operator_name}' does not match "
                        f"BoundaryOperatorSpec.name '{operator_spec.name}'"
                    )
                if spec.shape not in operator_spec.allowed_field_shapes:
                    raise ValueError(
                        f"PDE '{self.name}' boundary field '{name}' has shape "
                        f"'{spec.shape}', but operator '{operator_name}' only "
                        f"supports {operator_spec.allowed_field_shapes}."
                    )

        for name, spec in self.states.items():
            if name != spec.name:
                raise ValueError(
                    f"PDE '{self.name}' state key '{name}' does not match "
                    f"StateSpec.name '{spec.name}'"
                )
            invalid_couplings = (
                set(spec.stochastic_couplings) - _VALID_STOCHASTIC_COUPLINGS
            )
            if invalid_couplings:
                raise ValueError(
                    f"PDE '{self.name}' state '{name}' has unsupported stochastic "
                    f"couplings {sorted(invalid_couplings)}."
                )
            if (
                spec.shape != "scalar"
                and "saturating_self" in spec.stochastic_couplings
            ):
                raise ValueError(
                    f"PDE '{self.name}' state '{name}' uses "
                    "'saturating_self', but v1 only supports that coupling for "
                    "scalar states."
                )

        for name, spec in self.outputs.items():
            if name != spec.name:
                raise ValueError(
                    f"PDE '{self.name}' output key '{name}' does not match "
                    f"OutputSpec.name '{spec.name}'"
                )
            if spec.source_kind == "state":
                if spec.source_name not in self.states:
                    raise ValueError(
                        f"PDE '{self.name}' output '{name}' references unknown "
                        f"state '{spec.source_name}'"
                    )
                source_shape = self.states[spec.source_name].shape
                if source_shape != spec.shape:
                    raise ValueError(
                        f"PDE '{self.name}' output '{name}' shape '{spec.shape}' "
                        f"does not match source state '{spec.source_name}' shape "
                        f"'{source_shape}'"
                    )
            elif spec.source_kind != "derived":
                raise ValueError(
                    f"PDE '{self.name}' output '{name}' has unsupported "
                    f"source kind '{spec.source_kind}'"
                )

        for field_name in self.static_fields:
            if field_name not in self.outputs:
                raise ValueError(
                    f"PDE '{self.name}' static field '{field_name}' does not "
                    f"match any declared output. Expected one of "
                    f"{sorted(self.outputs)}."
                )

    def parameter_names(self) -> set[str]:
        return {parameter.name for parameter in self.parameters}

    def parameter_specs(self) -> dict[str, PDEParameter]:
        return {parameter.name: parameter for parameter in self.parameters}

    def validate_parameters(self, parameters: dict[str, float | int]) -> None:
        """Validate a concrete parameter dictionary against the PDE spec."""
        expected = self.parameter_names()
        actual = set(parameters)
        if actual != expected:
            raise ValueError(
                f"PDE '{self.name}' parameters must be {sorted(expected)}. "
                f"Got {sorted(actual)}."
            )
        for parameter in self.parameters:
            parameter.validate_value(
                parameters[parameter.name],
                context=f"PDE '{self.name}' parameter '{parameter.name}'",
            )

    def coefficient_names(self) -> list[str]:
        return list(self.coefficients.keys())

    def input_names(self) -> list[str]:
        return list(self.inputs.keys())

    def boundary_field_names(self) -> list[str]:
        return list(self.boundary_fields.keys())

    def output_names(self) -> list[str]:
        return list(self.outputs.keys())

    def expected_outputs(
        self, output_modes: dict[str, str], gdim: int
    ) -> list[ConcreteOutputSpec]:
        outputs: list[ConcreteOutputSpec] = []
        for output_name, output_spec in self.outputs.items():
            mode = output_modes[output_name]
            if mode == "hidden":
                continue
            if output_spec.shape == "scalar":
                outputs.append(
                    ConcreteOutputSpec(
                        name=output_name,
                        output_name=output_name,
                        source_name=output_spec.source_name,
                    )
                )
                continue

            for component in output_spec.component_labels(gdim):
                outputs.append(
                    ConcreteOutputSpec(
                        name=f"{output_name}_{component}",
                        output_name=output_name,
                        source_name=output_spec.source_name,
                        component=component,
                    )
                )
        return outputs

    def validate_dimension(self, gdim: int) -> None:
        if gdim not in self.supported_dimensions:
            raise ValueError(
                f"PDE '{self.name}' supports dimensions "
                f"{self.supported_dimensions}, not {gdim}D."
            )
