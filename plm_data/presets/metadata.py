"""Preset specifications and public extension contracts."""

from dataclasses import dataclass

_COMPONENT_LABELS = ("x", "y", "z")


def _component_labels(gdim: int) -> tuple[str, ...]:
    return _COMPONENT_LABELS[:gdim]


@dataclass(frozen=True)
class PDEParameter:
    """A configurable parameter of a PDE preset."""

    name: str
    description: str


@dataclass(frozen=True)
class InputSpec:
    """Config-facing input declaration for a preset."""

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
    """A solved state variable of a preset."""

    name: str
    shape: str

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        if self.shape != "vector":
            return ()
        return _component_labels(gdim)


@dataclass(frozen=True)
class BoundaryOperatorSpec:
    """Config-facing declaration for one named boundary operator."""

    name: str
    value_shape: str | None
    requires_pair_with: bool = False
    operator_parameter_names: tuple[str, ...] = ()
    description: str = ""


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
    """A config-selectable output exported by a preset."""

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
    """A concrete output array exported by a preset."""

    name: str
    output_name: str
    source_name: str
    component: str | None = None


@dataclass(frozen=True)
class PresetSpec:
    """Metadata and validation contract describing a PDE preset."""

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    inputs: dict[str, InputSpec]
    boundary_fields: dict[str, BoundaryFieldSpec]
    states: dict[str, StateSpec]
    outputs: dict[str, OutputSpec]
    steady_state: bool
    supported_dimensions: list[int]

    def __post_init__(self) -> None:
        for name, spec in self.inputs.items():
            if name != spec.name:
                raise ValueError(
                    f"Preset '{self.name}' input key '{name}' does not match "
                    f"InputSpec.name '{spec.name}'"
                )

        for name, spec in self.boundary_fields.items():
            if name != spec.name:
                raise ValueError(
                    f"Preset '{self.name}' boundary field key '{name}' does not "
                    f"match BoundaryFieldSpec.name '{spec.name}'"
                )
            if spec.shape not in {"scalar", "vector"}:
                raise ValueError(
                    f"Preset '{self.name}' boundary field '{name}' has unsupported "
                    f"shape '{spec.shape}'"
                )
            for operator_name, operator_spec in spec.operators.items():
                if operator_name != operator_spec.name:
                    raise ValueError(
                        f"Preset '{self.name}' boundary field '{name}' operator key "
                        f"'{operator_name}' does not match "
                        f"BoundaryOperatorSpec.name '{operator_spec.name}'"
                    )

        for name, spec in self.states.items():
            if name != spec.name:
                raise ValueError(
                    f"Preset '{self.name}' state key '{name}' does not match "
                    f"StateSpec.name '{spec.name}'"
                )

        for name, spec in self.outputs.items():
            if name != spec.name:
                raise ValueError(
                    f"Preset '{self.name}' output key '{name}' does not match "
                    f"OutputSpec.name '{spec.name}'"
                )
            if spec.source_kind == "state":
                if spec.source_name not in self.states:
                    raise ValueError(
                        f"Preset '{self.name}' output '{name}' references unknown "
                        f"state '{spec.source_name}'"
                    )
                source_shape = self.states[spec.source_name].shape
                if source_shape != spec.shape:
                    raise ValueError(
                        f"Preset '{self.name}' output '{name}' shape '{spec.shape}' "
                        f"does not match source state '{spec.source_name}' shape "
                        f"'{source_shape}'"
                    )
            elif spec.source_kind != "derived":
                raise ValueError(
                    f"Preset '{self.name}' output '{name}' has unsupported "
                    f"source kind '{spec.source_kind}'"
                )

    def parameter_names(self) -> set[str]:
        return {parameter.name for parameter in self.parameters}

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
                f"Preset '{self.name}' supports dimensions "
                f"{self.supported_dimensions}, not {gdim}D."
            )


SCALAR_STANDARD_BOUNDARY_OPERATORS = {
    "dirichlet": BoundaryOperatorSpec(
        name="dirichlet",
        value_shape="field",
        description="Strong value boundary condition.",
    ),
    "neumann": BoundaryOperatorSpec(
        name="neumann",
        value_shape="field",
        description="Natural flux boundary condition.",
    ),
    "robin": BoundaryOperatorSpec(
        name="robin",
        value_shape="field",
        operator_parameter_names=("alpha",),
        description="Scalar Robin boundary condition.",
    ),
    "periodic": BoundaryOperatorSpec(
        name="periodic",
        value_shape=None,
        requires_pair_with=True,
        description="Periodic side-pair constraint.",
    ),
}

VECTOR_STANDARD_BOUNDARY_OPERATORS = {
    "dirichlet": BoundaryOperatorSpec(
        name="dirichlet",
        value_shape="field",
        description="Strong vector boundary condition.",
    ),
    "neumann": BoundaryOperatorSpec(
        name="neumann",
        value_shape="field",
        description="Natural vector traction boundary condition.",
    ),
    "periodic": BoundaryOperatorSpec(
        name="periodic",
        value_shape=None,
        requires_pair_with=True,
        description="Periodic side-pair constraint.",
    ),
}

MAXWELL_BOUNDARY_OPERATORS = {
    "dirichlet": VECTOR_STANDARD_BOUNDARY_OPERATORS["dirichlet"],
    "periodic": VECTOR_STANDARD_BOUNDARY_OPERATORS["periodic"],
    "absorbing": BoundaryOperatorSpec(
        name="absorbing",
        value_shape="field",
        description="Absorbing Maxwell boundary condition.",
    ),
}
