"""Shared test fixtures."""

import ast
import sys
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

from plm_data.core.runtime_config import (
    BoundaryConditionConfig,
    DomainConfig,
    InputConfig,
    OutputConfig,
    SimulationConfig,
    TimeConfig,
)
from plm_data.core.solver_strategies import CONSTANT_LHS_SCALAR_SPD
from tests.runtime_helpers import (
    boundary_field_config,
    constant,
    direct_solver_config,
    output_fields,
    scalar_expr,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"
_MIRRORED_TESTS_ROOT = _TESTS_ROOT / "plm_data"
_WORKER_PUBLIC_FUNCTION_COVERAGE: set[str] = set()


class _PublicFunctionCoverage:
    def __init__(self) -> None:
        self._functions = _discover_public_top_level_functions()
        self.covered: set[str] = set()
        self._previous_profile = None

    @property
    def functions(self) -> set[str]:
        return set(self._functions)

    def start(self) -> None:
        self._previous_profile = sys.getprofile()
        sys.setprofile(self._profile)

    def stop(self) -> None:
        sys.setprofile(self._previous_profile)

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call":
            key = _function_key(
                frame.f_code.co_filename,
                frame.f_code.co_firstlineno,
                frame.f_code.co_name,
            )
            if key in self._functions:
                self.covered.add(key)
        if self._previous_profile is not None:
            self._previous_profile(frame, event, arg)


def _function_key(filename: str, lineno: int, name: str) -> str:
    path = Path(filename)
    try:
        relpath = path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        relpath = path
    return f"{relpath}:{lineno}:{name}"


def _public_function_first_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if not node.decorator_list:
        return node.lineno
    return min(node.lineno, *(decorator.lineno for decorator in node.decorator_list))


def _discover_public_top_level_functions() -> set[str]:
    functions = set()
    for path in sorted((_REPO_ROOT / "plm_data").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            functions.add(
                _function_key(
                    str(path),
                    _public_function_first_line(node),
                    node.name,
                )
            )
    return functions


def _is_full_suite_run(config: pytest.Config) -> bool:
    if not config.args:
        return True
    for arg in config.args:
        path = Path(arg)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.resolve() in {_TESTS_ROOT, _MIRRORED_TESTS_ROOT}:
            return True
    return False


def pytest_configure(config: pytest.Config) -> None:
    if config.option.collectonly:
        return
    if not _is_full_suite_run(config):
        return
    coverage = _PublicFunctionCoverage()
    coverage.start()
    config._plm_public_function_coverage = coverage


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    coverage = getattr(session.config, "_plm_public_function_coverage", None)
    if coverage is None:
        return

    coverage.stop()
    if hasattr(session.config, "workeroutput"):
        session.config.workeroutput["plm_public_function_coverage"] = sorted(
            coverage.covered
        )
        return

    if exitstatus != pytest.ExitCode.OK:
        return

    covered = set(coverage.covered) | _WORKER_PUBLIC_FUNCTION_COVERAGE
    missing = sorted(coverage.functions - covered)
    if not missing:
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_sep("=", "public top-level function coverage")
        terminal.line("The full test suite did not call these public functions:")
        for key in missing:
            terminal.line(f"  {key}")
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_testnodedown(node, error) -> None:
    _WORKER_PUBLIC_FUNCTION_COVERAGE.update(
        node.workeroutput.get("plm_public_function_coverage", ())
    )


@pytest.fixture
def rectangle_domain():
    return DomainConfig(
        type="rectangle",
        params={"size": [1.0, 1.0], "mesh_resolution": [8, 8]},
    )


@pytest.fixture
def direct_solver():
    return direct_solver_config(CONSTANT_LHS_SCALAR_SPD)


@pytest.fixture
def heat_config(tmp_path, rectangle_domain, direct_solver):
    return SimulationConfig(
        pde="heat",
        parameters={},
        domain=rectangle_domain,
        inputs={
            "u": InputConfig(
                source=scalar_expr("none"),
                initial_condition=scalar_expr(
                    "gaussian_bump",
                    sigma=0.1,
                    amplitude=1.0,
                    center=[0.5, 0.5],
                ),
            )
        },
        boundary_conditions={
            "u": boundary_field_config(
                {
                    "x-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "x+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y-": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                    "y+": BoundaryConditionConfig(type="neumann", value=constant(0.0)),
                }
            ),
        },
        output=OutputConfig(
            path=tmp_path,
            resolution=[4, 4],
            num_frames=2,
            formats=["numpy"],
            fields=output_fields(u="scalar"),
        ),
        solver=direct_solver,
        time=TimeConfig(dt=0.01, t_end=0.01),
        seed=42,
        coefficients={"kappa": constant(0.01)},
    )
