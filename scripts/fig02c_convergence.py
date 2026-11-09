"""Figure 2C: convergence of the best-so-far RMSE (median and IQR over 13 runs).

Output: figures/fig02c.pdf (and .png)
Data:   data/experimental_results/*_history_variant_AAV_run_*.csv
"""

import numpy as np
import matplotlib.pyplot as plt

import common as c

MAX_EVALS = 22000
# First evaluation, then every 2000 evaluations from 1000 to 21000
EVAL_POINTS = np.r_[1, np.arange(1000, 22001, 2000)]
DRAW_ORDER = ["cmaes", "lshade", "basin_hopping", "dual_annealing", "pso"]


def main() -> None:
    c.apply_publication_style(text_scale=1 / c.FIG2_PRINT_SCALE)
    num_points = len(c.load_experimental_trace())

    histories = {algo: [] for algo in c.ALGORITHMS}
    for algo, _, hist in c.iter_run_histories():
        best_so_far = np.minimum.accumulate(np.sqrt(hist["sse"].to_numpy(float) / num_points))
        padded = np.full(MAX_EVALS, np.nan)
        n = min(MAX_EVALS, len(best_so_far))
        padded[:n] = best_so_far[:n]
        histories[algo].append(padded)

    fig, ax = c.new_figure()
    for algo in DRAW_ORDER:
        sampled = np.vstack(histories[algo])[:, EVAL_POINTS - 1]
        median = np.nanmedian(sampled, axis=0)
        q1, q3 = np.nanpercentile(sampled, [25, 75], axis=0)
        color = c.COLOR_MAP[algo]
        ax.fill_between(EVAL_POINTS, q1, q3, step="post", color=color, alpha=0.16, linewidth=0)
        ax.step(EVAL_POINTS, median, where="post", color=color, linewidth=1.0)
        ax.plot(EVAL_POINTS[-1], median[-1], "o", color=color, markersize=3)

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("RMSE (a.u.)")
    c.align_ylabel(ax)
    c.tick_axis(ax, "x", 0, 21000, 7000)
    c.tick_axis(ax, "y", 0, 500, 100)  # 500 so the early IQR bands are not cut off
    c.add_panel_label(ax, "C")
    c.save_figure(fig, "fig02c")


if __name__ == "__main__":
    main()
