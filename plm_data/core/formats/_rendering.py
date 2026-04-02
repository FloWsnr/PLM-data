"""Shared matplotlib rendering for GIF and video output."""

from pathlib import Path

import numpy as np


def render_animation(
    data: np.ndarray,
    name: str,
    output_path: Path,
    writer_name: str,
    fps: int = 10,
) -> None:
    """Render a numpy array sequence as an animation.

    Args:
        data: Array of shape (num_frames, *resolution). 2D or 3D spatial data.
        name: Field name (used in title).
        output_path: Full path for the output file (e.g., field.gif or field.mp4).
        writer_name: Matplotlib animation writer ("pillow" for GIF, "ffmpeg" for video).
        fps: Frames per second.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    # Complex-valued fields (e.g. time-harmonic Maxwell): render magnitude.
    if np.iscomplexobj(data):
        data = np.abs(data)

    ndim = data.ndim - 1  # spatial dimensions (exclude frame axis)
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))

    if ndim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data[0].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(name)

        def update_2d(frame_idx: int) -> list:
            im.set_data(data[frame_idx].T)
            return [im]

        anim = animation.FuncAnimation(
            fig, update_2d, frames=len(data), blit=True, interval=1000 // fps
        )
    elif ndim == 3:
        # Render middle slices along each axis
        nx, ny, nz = data.shape[1], data.shape[2], data.shape[3]
        mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(name)

        # XY plane (z=mid_z), XZ plane (y=mid_y), YZ plane (x=mid_x)
        slices_0 = data[0]
        im_xy = axes[0].imshow(
            slices_0[:, :, mid_z].T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axes[0].set_title(f"XY (z={mid_z})")
        fig.colorbar(im_xy, ax=axes[0])

        im_xz = axes[1].imshow(
            slices_0[:, mid_y, :].T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axes[1].set_title(f"XZ (y={mid_y})")
        fig.colorbar(im_xz, ax=axes[1])

        im_yz = axes[2].imshow(
            slices_0[mid_x, :, :].T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axes[2].set_title(f"YZ (x={mid_x})")
        fig.colorbar(im_yz, ax=axes[2])

        def update_3d(frame_idx: int) -> list:
            s = data[frame_idx]
            im_xy.set_data(s[:, :, mid_z].T)
            im_xz.set_data(s[:, mid_y, :].T)
            im_yz.set_data(s[mid_x, :, :].T)
            return [im_xy, im_xz, im_yz]

        anim = animation.FuncAnimation(
            fig, update_3d, frames=len(data), blit=True, interval=1000 // fps
        )
    else:
        plt.close("all")
        raise ValueError(
            f"Cannot render animation for {ndim}D spatial data (expected 2D or 3D)"
        )

    fig.tight_layout()
    anim.save(str(output_path), writer=writer_name, fps=fps)
    plt.close(fig)
