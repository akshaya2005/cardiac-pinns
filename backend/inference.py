"""
Inference and ground-truth generation for the FHN cardiac PINN models.
"""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gifs import gif_from_grids
from schemas import PredictRequest, PredictResponse

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DATA_DIR = Path(__file__).resolve().parent / "data"

# matplotlib is case-sensitive about colormap names; the frontend's
# COLOR_SCHEME_OPTIONS values (src/constants.js) are lowercase/hyphenated.
CMAP_ALIASES = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "coolwarm": "coolwarm",
    "rdbu": "RdBu",
}


def resolve_cmap(color_scheme: str, default: str = "jet") -> str:
    return CMAP_ALIASES.get(color_scheme, default)


# Ground truth config
# Define the coordinates for the square's corners
x_min, y_min = 0, 0  # Lower-left corner
x_max, y_max = 1.0, 1.0  # Upper-right corner for a square with side length 1
t_min, t_max = 0, 10

# Generating Data
a = 0.1
gamma = 1
beta = 0.5
eps = 0.01
delta = 0.0


dx = 0.5
dt = 0.1

diff = 0.5  ## diffusion coefficient
endtime = 200  ## length of simulation in physical time units
nend = int(np.ceil(endtime / dt))  ## length of simulation in time steps
nx = 100  ##2d square length in gridpoints; physical length is nx * dx


# --- Network architectures -------------------------------------------------
# Mirrors the `network()` method in each backend/models/*.py file. Built
# directly here rather than instantiating the full FitzHughNagumo training
# wrapper classes, since those require ic/bc/data/res training points in
# their __init__ that aren't available (or needed) at inference time.


def _build_nn5(n_out: int = 2) -> nn.Module:
    """5 hidden layers of 60 — models/FHN_NN.py, FHN_PINN.py, FHN_Adaptive.py."""
    return nn.Sequential(
        nn.Linear(3, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, n_out),
    )


def _build_nn3(n_out: int = 2) -> nn.Module:
    """3 hidden layers of 60 — models/FHN_Time_Window.py."""
    return nn.Sequential(
        nn.Linear(3, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, n_out),
    )


# Maps the `model` value sent by the frontend (see
# pinn-visualization/src/constants.js -> MODEL_OPTIONS) to its architecture
# builder + weight file under /weights.
# "time-window" isn't listed here — it uses a separate net per time window
# (see predict_gif_time_window below) rather than a single weight file.
# TODO: confirm which file/weight "vanilla-pinn" actually uses.
MODEL_REGISTRY = {
    "nn": (_build_nn5, "modelNN.pt"),  # models/FHN_NN.py
    "adaptive-residual": (_build_nn5, "model3.pt"),  # models/FHN_Adaptive.py
    "vanilla-pinn": (_build_nn5, "modelVanillaPINN.pt"),
}

N_PRED_FRAMES = 50

# models/FHN_Time_Window.py weights, one net per marching window.
TIME_WINDOW_MODEL_FILES = [f"model_window_{i}.pt" for i in range(1, 10)]
TIME_WINDOW_FRAMES_PER_WINDOW = 8

_pred_x = np.linspace(x_min, x_max, nx)
_pred_y = np.linspace(y_min, y_max, nx)
_PRED_X, _PRED_Y = np.meshgrid(_pred_x, _pred_y, indexing="ij")
_X_FLAT = torch.tensor(_PRED_X.reshape(-1, 1), dtype=torch.float32)
_Y_FLAT = torch.tensor(_PRED_Y.reshape(-1, 1), dtype=torch.float32)


def load_model(model_key: str) -> nn.Module:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}")

    build_net, weight_file = MODEL_REGISTRY[model_key]
    if weight_file is None:
        raise NotImplementedError(f"No weight file configured for '{model_key}' yet")

    net = build_net()
    state_dict = torch.load(WEIGHTS_DIR / weight_file, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    return net


def predict_gif(model_key: str, color_scheme: str) -> str:
    """Run a net over the domain at a sequence of time steps and assemble a GIF.

    Mirrors /reference's generate_u_gif, but calls the net directly (the
    pasted model classes only define __init__ + network(), no forward/
    function method yet) and returns an in-memory data URI instead of
    writing a file.
    """
    if model_key == "time-window":
        return predict_gif_time_window(color_scheme)

    net = load_model(model_key)

    t_values = np.linspace(t_min, t_max, N_PRED_FRAMES)
    grids = []

    with torch.no_grad():
        for t_val in t_values:
            t_flat = torch.full((nx * nx, 1), float(t_val), dtype=torch.float32)
            inputs = torch.cat([_X_FLAT, _Y_FLAT, t_flat], dim=1)
            u_pred = net(inputs)[:, 0:1]
            grids.append(u_pred.numpy().reshape(nx, nx))

    return gif_from_grids(grids, t_values, cmap=resolve_cmap(color_scheme))


def predict_gif_time_window(color_scheme: str) -> str:
    """Time-window marching: load a fresh net per window, stitched into one GIF.

    Adapted from /reference's multi-window generate_u_gif — loads
    TIME_WINDOW_MODEL_FILES in order, each covering an equal slice of
    [t_min, t_max], and renders TIME_WINDOW_FRAMES_PER_WINDOW frames per
    window. TODO: the real window boundaries (`times` in the reference
    code) weren't available here, so this assumes evenly spaced windows —
    swap in the actual boundaries if they're non-uniform.
    """
    times = np.linspace(t_min, t_max, len(TIME_WINDOW_MODEL_FILES) + 1)

    grids = []
    t_values = []

    for i, weight_file in enumerate(TIME_WINDOW_MODEL_FILES):
        net = _build_nn3()
        state_dict = torch.load(WEIGHTS_DIR / weight_file, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()

        t_start, t_end = times[i], times[i + 1]

        with torch.no_grad():
            for t_val in np.linspace(t_start, t_end, TIME_WINDOW_FRAMES_PER_WINDOW):
                t_flat = torch.full((nx * nx, 1), float(t_val), dtype=torch.float32)
                inputs = torch.cat([_X_FLAT, _Y_FLAT, t_flat], dim=1)
                u_pred = net(inputs)[:, 0:1]
                grids.append(u_pred.numpy().reshape(nx, nx))
                t_values.append(t_val)

    return gif_from_grids(grids, t_values, cmap=resolve_cmap(color_scheme))


def _placeholder_image(seed: int, cmap: str = "gray") -> str:
    rng = np.random.default_rng(seed)
    data = rng.random((50, 50))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(data, cmap=cmap)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


N_GT_FRAMES = 50  # matches N_PRED_FRAMES so both GIFs render at the same pace
GT_SAVE_EVERY = nend // N_GT_FRAMES


def generate_ground_truth(initial_condition: str):
    """Numerically solve the FHN system for the chosen initial condition.

    TODO: `initial_condition` isn't used yet — this always runs the
    corner/circular stimulus below. Add branches here once more initial
    conditions are needed.
    """

    u = np.zeros((nx, nx), dtype=float)  ##fast "voltage"-like variable
    v = np.zeros((nx, nx), dtype=float)  ##slow recovery variable

    radius = nx // 4

    x = np.arange(nx)
    y = np.arange(nx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    mask = X**2 + Y**2 <= radius**2
    u[mask] = 0.5

    v1 = v.copy()
    u1 = u.copy()
    training_points = np.zeros((N_GT_FRAMES, nx, nx))
    training_points_v = np.zeros((N_GT_FRAMES, nx, nx))

    ## Generating data
    for ntime in range(nend):  # 500 is the number of timesteps

        uold = u1.copy()
        vold = v1.copy()
        dv1 = eps * (beta * uold - gamma * vold - delta)
        du1 = (a - uold) * (uold - 1) * uold - vold

        diff_x = np.zeros((nx, nx), dtype=float)
        diff_y = np.zeros((nx, nx), dtype=float)
        diff_x[0, :] = 2 * (uold[2, :] - uold[1, :])
        diff_x[nx - 1, :] = 2 * (-uold[nx - 1, :] + uold[nx - 2, :])
        diff_x[1:-1, :] = uold[0:-2, :] - 2 * uold[1:-1, :] + uold[2:, :]

        diff_y[:, 0] = 2 * (uold[:, 2] - uold[:, 1])
        diff_y[:, nx - 1] = 2 * (-uold[:, nx - 1] + uold[:, nx - 2])
        diff_y[:, 1:-1] = uold[:, 0:-2] - 2 * uold[:, 1:-1] + uold[:, 2:]

        u1 = uold + dt * du1 + diff * dt / (dx * dx) * (diff_x + diff_y)
        v1 = vold + dt * dv1

        if ntime % GT_SAVE_EVERY == 0:
            frame_idx = ntime // GT_SAVE_EVERY
            if frame_idx < N_GT_FRAMES:
                training_points[frame_idx, :, :] = u1.T
                training_points_v[frame_idx, :, :] = v1.T

    return training_points


def get_ground_truth(initial_condition: str) -> np.ndarray:
    """FD-solve the FHN system once per initial condition and cache the
    result to a CSV under backend/data/, so repeat requests just load it
    instead of re-running the solver."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"ground_truth_{initial_condition}.csv"

    if cache_path.exists():
        flat = np.loadtxt(cache_path, delimiter=",")
        return flat.reshape(N_GT_FRAMES, nx, nx)

    training_points = generate_ground_truth(initial_condition)
    np.savetxt(cache_path, training_points.reshape(N_GT_FRAMES, -1), delimiter=",", fmt="%.6f")
    return training_points


def ground_truth_gif(initial_condition: str, color_scheme: str) -> str:
    training_points = get_ground_truth(initial_condition)
    # Label frames on a 0-10 scale (endtime / 20) to match the network
    # prediction's t_min/t_max, even though the solver itself still runs
    # the full endtime=200 physical duration.
    t_values = np.linspace(0, endtime / 20, training_points.shape[0])
    return gif_from_grids(list(training_points), t_values, cmap=resolve_cmap(color_scheme))


def run_prediction(request: PredictRequest) -> PredictResponse:
    try:
        network_gif = predict_gif(request.model, request.color_scheme)
    except (NotImplementedError, ValueError, FileNotFoundError):
        # Model/weight not wired up yet — fall back to a placeholder so the
        # endpoint still returns something displayable.
        network_gif = _placeholder_image(seed=1, cmap="gray")

    ground_truth = ground_truth_gif(request.initial_condition, request.color_scheme)

    return PredictResponse(
        network_image=network_gif,
        ground_truth_image=ground_truth,
        training_points=None,
        accuracy=None,
    )
