"""Shared paths, style, data loading, and model code for the paper figures."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTAL_RUNS_DIR = DATA_DIR / "experimental_results"
TARGETS_DIR = DATA_DIR / "fitting_targets_and_references"
SYNTHETIC_RESULTS_CSV = DATA_DIR / "synthetic_results.csv"
FIGURES_DIR = REPO_ROOT / "figures"

VARIANT = "AAV"
NUM_RUNS = 13

# ---------------------------------------------------------------------------
# Optimizer names and colors (shared by every figure)
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "basin_hopping": "Basin Hopping",
    "cmaes": "CMA-ES",
    "dual_annealing": "Dual Annealing",
    "lshade": "L-SHADE",
    "pso": "PSO",
}
ALGORITHMS = list(DISPLAY_NAMES.keys())
COLOR_MAP = {
    "basin_hopping": "#FFBE0B",
    "cmaes": "#FB5607",
    "dual_annealing": "#FF006E",
    "lshade": "#8338EC",
    "pso": "#3A86FF",
}
DISPLAY_COLOR_MAP = {DISPLAY_NAMES[a]: COLOR_MAP[a] for a in ALGORITHMS}

# ---------------------------------------------------------------------------
# Publication style, at true printed size. Every figure has the same plotting area
# (the box inside the axes) and the same text sizes; only the margins around the box
# (tick labels, axis labels, legends) differ. Include the PNGs in LaTeX at their
# natural size (scale 1), except Figure 2, whose three panels are scaled down to fit
# across a 7 in (177.8 mm) ACS page.
# ---------------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
MM_TO_PT = 72.0 / 25.4
SINGLE_COLUMN_MM = 82.5       # ACS one-column figure (3.25 in)
DOUBLE_COLUMN_MM = 177.8      # ACS full-width figure (7 in)
AXES_MM = (55.0, 41.0)        # plotting area (width, height) of every figure
YLABEL_OFFSET_MM = 10.0       # y-axis label distance from the box, same in every plot
DPI = 600

FONT_SIZE = 7                 # tick labels, legends, numbers inside plots
LABEL_SIZE = 8                # axis labels
PANEL_LABEL_SIZE = 10         # A, B, C

# Figure 2's three panels are shrunk by FIG2_PRINT_SCALE to fit across the page, so
# their text is drawn larger by 1 / FIG2_PRINT_SCALE and prints at the sizes above.
# Legends are left at FONT_SIZE. Include each panel with
# \includegraphics[scale=FIG2_PRINT_SCALE]. Solved so the three panels plus two 4 mm
# gaps fill the 177.8 mm page; re-solve if Figure 2's layout changes.
FIG2_PRINT_SCALE = 0.744
_text_scale = 1.0             # set by apply_publication_style


def new_figure(axes_mm: tuple[float, float] = AXES_MM) -> tuple[plt.Figure, plt.Axes]:
    """Figure whose plotting area is exactly axes_mm (width, height) in mm.

    The canvas starts with wide margins; save_figure crops them to the content, which
    never changes the size of the plotting area.
    """
    margin = 35.0
    width, height = axes_mm
    fig_w, fig_h = width + 2 * margin, height + 2 * margin
    fig = plt.figure(figsize=(fig_w * MM_TO_INCH, fig_h * MM_TO_INCH))
    ax = fig.add_axes((margin / fig_w, margin / fig_h, width / fig_w, height / fig_h))
    return fig, ax


def align_ylabel(ax: plt.Axes) -> None:
    """Put the y-axis label at the same distance from the box in every plot."""
    width_mm = ax.get_position().width * ax.figure.get_figwidth() / MM_TO_INCH
    ax.yaxis.set_label_coords(-YLABEL_OFFSET_MM * _text_scale / width_mm, 0.5)


def apply_publication_style(text_scale: float = 1.0) -> None:
    """Set the rcParams shared by every paper figure.

    text_scale enlarges all text except legends (Figure 2 uses 1 / FIG2_PRINT_SCALE).
    """
    global _text_scale
    _text_scale = text_scale
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FONT_SIZE * text_scale,
        "axes.labelsize": LABEL_SIZE * text_scale,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE * text_scale,
        "ytick.labelsize": FONT_SIZE * text_scale,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": True,  # full box around every plot
        "axes.spines.right": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tick_axis(ax: plt.Axes, axis: str, start: float, stop: float, step: float) -> None:
    """Make a numeric axis run exactly from its first tick to its last tick."""
    ticks = np.round(np.arange(start, stop + step / 2, step), 10)
    if axis == "x":
        ax.set_xlim(start, stop)
        ax.set_xticks(ticks)
    else:
        ax.set_ylim(start, stop)
        ax.set_yticks(ticks)


def style_optimizer_xticks(ax: plt.Axes, labels: list[str]) -> None:
    """Rotate optimizer names on the x-axis the same way in every figure."""
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right", rotation_mode="anchor")


def violin_plot(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, order: list[str],
                palette: dict, median_fmt: str, median_size: float = FONT_SIZE) -> None:
    """Violins with a light fill and solid outline in each optimizer's color.

    No individual points. The solid black line inside each violin is the median,
    whose value is printed just above the violin at median_size pt (printed size).
    """
    n_lines_before, n_colls_before = len(ax.lines), len(ax.collections)
    sns.violinplot(ax=ax, data=data, x=x, y=y, order=order, hue=x, palette=palette,
                   legend=False, saturation=1, inner="quart", cut=0,
                   density_norm="width", width=0.75, linewidth=1.0)
    for coll in ax.collections[n_colls_before:]:
        r, g, b, _ = coll.get_facecolor()[0]
        coll.set_facecolor((r, g, b, 0.35))
        coll.set_edgecolor((r, g, b, 1.0))
    # seaborn draws three dashed lines per violin (Q1, median, Q3); keep only the
    # median, drawn solid
    for k, line in enumerate(ax.lines[n_lines_before:]):
        if k % 3 == 1:
            line.set_linestyle("-")
            line.set_linewidth(1.4)
            line.set_color("black")
        else:
            line.remove()
    ax.set_xlabel("")  # no x-axis title: the optimizer names on the ticks say it
    medians = data.groupby(x)[y].median()
    tops = data.groupby(x)[y].max()
    for i, key in enumerate(order):
        ax.annotate(format(float(medians[key]), median_fmt), xy=(i, float(tops[key])),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=median_size * _text_scale, color="black")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Bold panel letter (A, B, C) in the top-left corner of the panel.

    Call it last, just before saving. The letter sits just above the box, at the left
    edge of everything drawn (the y-axis label, or long tick labels in Figure 4A).
    """
    fig = ax.figure
    fig.canvas.draw()
    left_in = fig.get_tightbbox(fig.canvas.get_renderer()).x0
    # Offset from the box's left edge in points, so cropping on save can't move it
    dx_pt = (left_in - ax.get_position().x0 * fig.get_figwidth()) * 72
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=PANEL_LABEL_SIZE * _text_scale, fontweight="bold")


def save_figure(fig: plt.Figure, name: str, extra_artists: list | None = None) -> Path:
    """Save ``figures/<name>.pdf`` and ``figures/<name>.png``, cropped to their content,
    at true printed size.

    The PDF is vector (sharp at any zoom); the PNG (600 dpi) is for quick viewing.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.pdf"
    for path in (out, out.with_suffix(".png")):
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                    bbox_extra_artists=extra_artists,  # e.g. extra legends, kept in the crop
                    metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    print(f"Saved {out.relative_to(REPO_ROOT)} (+ .png)")
    return out


# ---------------------------------------------------------------------------
# Experimental data and run histories
# ---------------------------------------------------------------------------
def load_experimental_trace() -> np.ndarray:
    """GFP measurements (43 points, 0-420 min every 10 min) for the AAV variant."""
    return loadmat(TARGETS_DIR / "experimental_data.mat")[VARIANT].ravel().astype(float)


def iter_run_histories():
    """Yield (algorithm, run_index, history DataFrame) for all 65 experimental runs."""
    for algo in ALGORITHMS:
        for run_idx in range(1, NUM_RUNS + 1):
            path = EXPERIMENTAL_RUNS_DIR / f"{algo}_history_variant_{VARIANT}_run_{run_idx}.csv"
            if path.exists():
                yield algo, run_idx, pd.read_csv(path, usecols=["sse"])


def load_best_params(algo: str) -> np.ndarray:
    """Best parameter vector found by ``algo`` on the experimental data."""
    df = pd.read_csv(TARGETS_DIR / f"{algo}_best_params.csv")
    return df.iloc[0].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# ODE model (11 states, 21 parameters) used to draw the best fits
# ---------------------------------------------------------------------------
MODEL_PARAMS = {"P_x": 1e-9, "P_y": 1e-9, "P_z": 1e-9}
ATC_CONVERSION = 0.46822
ATC_NG_PER_ML = 50
SIM_DURATION_MIN = 420
SAMPLE_INTERVAL_MIN = 10


def _derivatives(t: float, state: np.ndarray, p: np.ndarray) -> np.ndarray:
    (STAR, THS, TetR, aTc, tetR_aTc, Y, Y_active, Pz_repressed, Pz_active, Z, GFP) = state
    free_pz = MODEL_PARAMS["P_z"] - Pz_repressed - Pz_active
    d = np.zeros(11)
    d[0] = MODEL_PARAMS["P_x"] * p[0] - p[11] * STAR * free_pz - p[5] * STAR + p[12] * Pz_active
    d[1] = MODEL_PARAMS["P_x"] * p[0] - p[13] * THS * Y - p[6] * THS
    d[2] = (
        p[1] * Y_active
        - p[14] * TetR * free_pz
        - p[15] * TetR * aTc
        - p[7] * TetR
        + p[16] * Pz_repressed
        + p[17] * tetR_aTc
        - p[20] * TetR * Pz_active
    )
    d[3] = -p[15] * TetR * aTc + p[17] * tetR_aTc
    d[4] = p[15] * TetR * aTc - p[17] * tetR_aTc
    d[5] = p[2] * MODEL_PARAMS["P_y"] - p[13] * THS * Y - p[8] * Y
    d[6] = p[13] * THS * Y - p[19] * Y_active
    d[7] = p[14] * TetR * free_pz - p[16] * Pz_repressed + p[20] * TetR * Pz_active
    d[8] = p[11] * STAR * free_pz - p[12] * Pz_active - p[20] * TetR * Pz_active
    d[9] = p[3] * Pz_active - p[9] * Z
    d[10] = p[4] * Z - p[10] * GFP
    return d


def simulate_gfp(params: np.ndarray) -> np.ndarray:
    """Simulated GFP fluorescence (a.u.) at the 43 measurement times."""
    p = np.asarray(params, dtype=float).copy()
    p[7] *= 2  # AAV tag doubles TetR degradation
    state0 = np.zeros(11)
    state0[3] = (ATC_NG_PER_ML / ATC_CONVERSION) * 1e-9
    t_eval = np.arange(0, SIM_DURATION_MIN + SAMPLE_INTERVAL_MIN, SAMPLE_INTERVAL_MIN) * 60.0
    sol = solve_ivp(
        lambda t, y: _derivatives(t, y, p),
        [t_eval[0], t_eval[-1]],
        state0,
        t_eval=t_eval,
        method="BDF",
        atol=1e-11,
        rtol=1e-11,
    )
    return sol.y[10] * 1e9 * 10 ** p[18]


# ---------------------------------------------------------------------------
# Synthetic benchmark (750 runs) and parameter recovery (MASE)
# ---------------------------------------------------------------------------
LOWER_BOUNDS = np.array([
    0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
    0.0001, 0.0001, 0.00005, 0.0001, 0.0001, 0.00005,
    100, 1e-8, 100, 100, 100, 1e-8, 1e-8, 0.5, 0.0001, 100,
])
UPPER_BOUNDS = np.array([
    0.5, 0.5, 0.5, 0.5, 0.5,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    1_000_000, 1e-5, 1_000_000, 1_000_000, 1_000_000, 1e-5, 1e-5, 5, 0.01, 1_000_000,
])


def synthetic_dataset_length() -> int:
    """Number of time points in each synthetic dataset (43)."""
    return len(pd.read_csv(TARGETS_DIR / "basin_hopping_1.csv"))


def mase(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error after scaling every parameter to [0, 1] by its bounds."""
    scale = UPPER_BOUNDS - LOWER_BOUNDS
    return float(np.mean(np.abs((found - LOWER_BOUNDS) / scale - (truth - LOWER_BOUNDS) / scale)))


def load_synthetic_results() -> pd.DataFrame:
    """All 750 synthetic runs with RMSE, MASE, and dataset family ("category")."""
    df = pd.read_csv(SYNTHETIC_RESULTS_CSV)
    df["category"] = df["dataset"].str.rsplit("_", n=1).str[0]
    df["rmse"] = (df["min_sse"] / synthetic_dataset_length()) ** 0.5
    truth = {c: load_best_params(c) for c in df["category"].unique()}
    df["mase"] = [
        mase(np.asarray(ast.literal_eval(params), dtype=float), truth[cat])
        for params, cat in zip(df["params"], df["category"])
    ]
    df["algorithm"] = df["optimizer"].map(DISPLAY_NAMES)
    return df
