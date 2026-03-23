"""Data classes for PDE preset metadata."""

from dataclasses import dataclass


@dataclass
class PDEParameter:
    """A configurable parameter of a PDE preset."""

    name: str
    description: str


@dataclass
class PDEMetadata:
    """Metadata describing a PDE preset."""

    name: str
    category: str
    description: str
    equations: dict[str, str]
    parameters: list[PDEParameter]
    field_names: list[str]
    steady_state: bool
    supported_dimensions: list[int]
