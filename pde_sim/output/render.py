"""Rendering helpers used by output handlers.

We intentionally avoid creating matplotlib figures for per-frame rendering to keep
outputs fast and headless-friendly.
"""

import matplotlib.pyplot as plt
import numpy as np


def render_colormap_rgb(
    data: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
    colormap: str,
) -> np.ndarray:
    """Convert a 2D array into an uint8 RGB image using a matplotlib colormap.

    The transformation matches the previous behavior used in the project:
    - transpose (so the first axis is shown as x)
    - flip vertically to emulate `origin="lower"`
    """

    if np.iscomplexobj(data):
        data = np.abs(data)

    if vmin is None or vmax is None:
        vmin = float(np.min(data))
        vmax = float(np.max(data))

    # Avoid division-by-zero and non-finite ranges (constant fields are valid).
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        normalized = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = (data.T - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)

    cmap = plt.get_cmap(colormap)
    rgba = cmap(normalized)
    rgb = (rgba[::-1, :, :3] * 255).astype(np.uint8)
    return rgb

