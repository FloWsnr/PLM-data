"""Shared airfoil profile helpers for Gmsh-backed channel domains."""

import math

import numpy as np


def _rotation_matrix(angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=float,
    )


def symmetric_naca_airfoil_surfaces(
    *,
    chord_length: float,
    thickness_ratio: float,
    center: np.ndarray | list[float] | tuple[float, float],
    attack_angle_degrees: float,
    num_chord_samples: int = 81,
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper and lower point clouds for a symmetric NACA 00xx airfoil."""
    if num_chord_samples < 3:
        raise ValueError("Airfoil sampling requires at least 3 chord samples.")

    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,):
        raise ValueError(
            "Airfoil center must contain exactly two coordinates. "
            f"Got shape {center_array.shape}."
        )

    beta = np.linspace(0.0, math.pi, num_chord_samples, dtype=float)
    xi = 0.5 * (1.0 - np.cos(beta))
    half_thickness = (
        5.0
        * thickness_ratio
        * (
            0.2969 * np.sqrt(np.clip(xi, 0.0, 1.0))
            - 0.1260 * xi
            - 0.3516 * xi**2
            + 0.2843 * xi**3
            - 0.1015 * xi**4
        )
    )

    upper = np.column_stack(
        (
            (xi[::-1] - 0.5) * chord_length,
            half_thickness[::-1] * chord_length,
        )
    )
    lower = np.column_stack(
        (
            (xi[1:] - 0.5) * chord_length,
            -half_thickness[1:] * chord_length,
        )
    )

    rotation = _rotation_matrix(attack_angle_degrees)
    return (
        upper @ rotation.T + center_array[None, :],
        lower @ rotation.T + center_array[None, :],
    )


def symmetric_naca_airfoil_outline(
    *,
    chord_length: float,
    thickness_ratio: float,
    center: np.ndarray | list[float] | tuple[float, float],
    attack_angle_degrees: float,
    num_chord_samples: int = 81,
) -> np.ndarray:
    """Return one closed-outline point cloud for a symmetric NACA 00xx airfoil."""
    upper, lower = symmetric_naca_airfoil_surfaces(
        chord_length=chord_length,
        thickness_ratio=thickness_ratio,
        center=center,
        attack_angle_degrees=attack_angle_degrees,
        num_chord_samples=num_chord_samples,
    )
    return np.vstack((upper, lower))
