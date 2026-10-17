"""
Shared GIF-assembly utilities, adapted from the `generate_u_gif` pattern in
/reference. Instead of writing a .gif file to disk, frames are assembled
in-memory and returned as a base64 data URI so the API can hand them
straight to the frontend's <img> tag.
"""

import base64
import io

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def frames_to_gif_data_uri(frames, fps: int = 8) -> str:
    buf = io.BytesIO()
    imageio.mimwrite(buf, frames, format="GIF", fps=fps)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/gif;base64,{encoded}"


def grid_frame(u_grid: np.ndarray, t_label: float, cmap: str, levels: int = 30) -> np.ndarray:
    """Render one u(x, y) snapshot to an RGB array, same look as generate_u_gif."""
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contourf(u_grid, levels=levels, cmap=cmap)
    ax.set_title(f"u(x, y, t), t = {t_label:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(contour)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    image = buf[..., :3].copy()
    plt.close(fig)
    return image


def gif_from_grids(grids, t_values, cmap: str = "jet", fps: int = 8) -> str:
    """Turn a sequence of u(x, y) snapshots + matching time labels into a GIF."""
    frames = [grid_frame(grid, t_val, cmap) for grid, t_val in zip(grids, t_values)]
    return frames_to_gif_data_uri(frames, fps=fps)
