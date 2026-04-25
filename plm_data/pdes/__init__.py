"""PDE registry and runtime contracts."""

from plm_data.pdes.base import (
    CustomProblem,
    PDE,
    ProblemInstance,
    RunResult,
    TransientLinearProblem,
    TransientNonlinearProblem,
)
from plm_data.pdes.registry import get_pde, list_pdes, register_pde
from plm_data.pdes.metadata import (
    BoundaryFieldSpec,
    CoefficientSpec,
    ConcreteOutputSpec,
    InputSpec,
    OutputSpec,
    PDEParameter,
    PDEParameterSampler,
    PDESpec,
    StateSpec,
)

__all__ = [
    "BoundaryFieldSpec",
    "CoefficientSpec",
    "ConcreteOutputSpec",
    "CustomProblem",
    "InputSpec",
    "OutputSpec",
    "PDE",
    "PDEParameter",
    "PDEParameterSampler",
    "PDESpec",
    "ProblemInstance",
    "RunResult",
    "StateSpec",
    "TransientLinearProblem",
    "TransientNonlinearProblem",
    "get_pde",
    "list_pdes",
    "register_pde",
]
