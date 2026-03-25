"""Preset specifications and public extension contracts."""

from dataclasses import dataclass

_COMPONENT_LABELS = ("x", "y", "z")


@dataclass(frozen=True)
class PDEParameter:
    """A configurable parameter of a PDE preset."""

    name: str
    description: str


@dataclass(frozen=True)
class FieldSpec:
    """Config-facing field declaration for a preset."""

    name: str
    shape: str
    allow_boundary_conditions: bool
    allow_source: bool
    allow_initial_condition: bool
    output_mode: str

    def component_labels(self, gdim: int) -> tuple[str, ...]:
        """Return active component labels for the given dimension."""
        if self.shape != "vector":
            return ()
        return _COMPONENT_LABELS[:gdim]

    def output_names(self, gdim: int, mode: str) -> list[str]:
        """Return concrete output array names for the requested mode."""
        if mode == "hidden":
            return []
        if self.shape == "scalar":
            if mode != "scalar":
                raise ValueError(
                    f"Scalar field '{self.name}' does not support output mode '{mode}'"
                )
            return [self.name]
        if mode != "components":
            raise ValueError(
                f"Vector field '{self.name}' does not support output mode '{mode}'"
            )
        return [f"{self.name}_{label}" for label in self.component_labels(gdim)]

    def validate_output_mode(self, mode: str) -> None:
        """Validate a config-selected output mode against this field spec."""
        allowed = {self.output_mode, "hidden"}
        if mode not in allowed:
            raise ValueError(
                f"Field '{self.name}' output mode must be one of {sorted(allowed)}. "
                f"Got '{mode}'."
            )


@dataclass(frozen=True)
class OutputFieldSpec:
    """A concrete output array exported by a preset."""

    name: str
    field_name: str
    component: str | None = None


@dataclass(frozen=True)
class PresetSpec:
    """Metadata and validation contract describing a PDE preset."""

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    fields: dict[str, FieldSpec]
    family: str
    steady_state: bool
    supported_dimensions: list[int]

    def parameter_names(self) -> set[str]:
        """Return configured parameter names."""
        return {parameter.name for parameter in self.parameters}

    def expected_outputs(
        self, output_modes: dict[str, str], gdim: int
    ) -> list[OutputFieldSpec]:
        """Return the concrete output arrays for a config in the given dimension."""
        outputs: list[OutputFieldSpec] = []
        for field_name, field_spec in self.fields.items():
            mode = output_modes[field_name]
            if mode == "hidden":
                continue
            if field_spec.shape == "scalar":
                outputs.append(OutputFieldSpec(name=field_name, field_name=field_name))
                continue
            for component in field_spec.component_labels(gdim):
                outputs.append(
                    OutputFieldSpec(
                        name=f"{field_name}_{component}",
                        field_name=field_name,
                        component=component,
                    )
                )
        return outputs

    def field_names(self) -> list[str]:
        """Return config-facing field names in declaration order."""
        return list(self.fields.keys())

    def validate_dimension(self, gdim: int) -> None:
        """Raise if the preset does not support the requested spatial dimension."""
        if gdim not in self.supported_dimensions:
            raise ValueError(
                f"Preset '{self.name}' supports dimensions "
                f"{self.supported_dimensions}, not {gdim}D."
            )
