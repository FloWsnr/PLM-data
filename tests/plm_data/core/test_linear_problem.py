"""Tests for plm_data.core.linear_problem."""

from types import SimpleNamespace

import plm_data.core.linear_problem as linear_problem
from plm_data.core.linear_problem import configure_nested_problem


class _FakePETSc:
    class Vec:
        class Type:
            NEST = "nest"

    class Mat:
        class Type:
            NEST = "nest"

        class Option:
            SPD = "spd"

    class NullSpace:
        def create(self, *, vectors):
            return SimpleNamespace(vectors=vectors)


class _FakeSubVector:
    def __init__(self):
        self.values: list[float] = []

    def set(self, value):
        self.values.append(value)


class _FakeNestedVector:
    def __init__(self, vector_type="nest"):
        self._type = vector_type
        self.sub_vectors = [_FakeSubVector(), _FakeSubVector()]
        self.normalized = False

    def getType(self):
        return self._type

    def duplicate(self):
        return self

    def getNestSubVecs(self):
        return self.sub_vectors

    def normalize(self):
        self.normalized = True


class _FakeSubMatrix:
    def __init__(self):
        self.options: list[tuple[str, bool]] = []

    def setOption(self, option, value):
        self.options.append((option, value))


class _FakeNestedMatrix:
    def __init__(self, matrix_type="nest"):
        self._type = matrix_type
        self.nullspace = None
        self.blocks = {
            (0, 0): _FakeSubMatrix(),
            (1, 1): _FakeSubMatrix(),
        }

    def getType(self):
        return self._type

    def setNullSpace(self, nullspace):
        self.nullspace = nullspace

    def getNestSubMatrix(self, row, column):
        return self.blocks[(row, column)]


def test_configure_nested_problem_sets_nullspace_and_spd_hints(monkeypatch):
    monkeypatch.setattr(linear_problem, "PETSc", _FakePETSc)
    A = _FakeNestedMatrix()
    P_mat = _FakeNestedMatrix()
    b = _FakeNestedVector()
    problem = SimpleNamespace(A=A, P_mat=P_mat, b=b, x=_FakeNestedVector())

    configure_nested_problem(
        problem,
        pressure_block=1,
        system_spd_blocks=(0,),
        preconditioner_spd_blocks=(1,),
    )

    assert A.nullspace.vectors == [b]
    assert b.normalized is True
    assert b.sub_vectors[0].values == [0.0]
    assert b.sub_vectors[1].values == [0.0, 1.0]
    assert A.blocks[(0, 0)].options == [("spd", True)]
    assert P_mat.blocks[(1, 1)].options == [("spd", True)]
