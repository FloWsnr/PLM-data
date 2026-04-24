"""Swift-Hohenberg PDE spec."""

from plm_data.pdes.swift_hohenberg.pde import SwiftHohenbergPDE

PDE_SPEC = SwiftHohenbergPDE().spec

__all__ = ["PDE_SPEC"]
