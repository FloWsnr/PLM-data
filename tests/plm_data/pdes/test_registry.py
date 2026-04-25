"""Tests for plm_data.pdes.registry."""

import plm_data.pdes.registry as pde_registry
from plm_data.pdes.registry import register_pde


def test_register_pde_adds_class_to_registry(monkeypatch):
    registry: dict[str, type] = {}
    monkeypatch.setattr(pde_registry, "_PDE_REGISTRY", registry)

    class DummyPDE:
        pass

    registered = register_pde("dummy")(DummyPDE)

    assert registered is DummyPDE
    assert registry == {"dummy": DummyPDE}
