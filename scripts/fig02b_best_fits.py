"""Figure 2B: best-fit GFP trajectory of each optimizer vs the experimental data.

Output: figures/fig02b.pdf (and .png)
Data:   data/fitting_targets_and_references/experimental_data.mat
        data/fitting_targets_and_references/<optimizer>_best_params.csv
"""

import numpy as np
import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style(text_scale=1 / c.FIG2_PRINT_SCALE)
    exp_trace = c.load_experimental_trace()
    time_min = np.arange(0, c.SIM_DURATION_MIN + c.SAMPLE_INTERVAL_MIN, c.SAMPLE_INTERVAL_MIN)

    sims = {algo: c.simulate_gfp(c.load_best_params(algo)) for algo in c.ALGORITHMS}
    rmse = {algo: float(np.sqrt(np.mean((exp_trace - sim) ** 2))) for algo, sim in sims.items()}
    order = sorted(c.ALGORITHMS, key=rmse.get)  # legend lists best fit first

    fig, ax = c.new_figure()
    ax.plot(time_min, exp_trace, "o", color="black", alpha=0.75, markersize=3,
            label="Experimental")
    for algo in order:
        ax.plot(time_min, sims[algo], color=c.COLOR_MAP[algo], linewidth=1.0,
                label=c.DISPLAY_NAMES[algo])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("GFP (a.u.)")
    c.align_ylabel(ax)
    c.tick_axis(ax, "x", 0, 420, 105)
    c.tick_axis(ax, "y", 0, 1000, 200)
    # Optimizer colors are keyed in the legend of Figure 2C; only the data marker here
    handles, labels = ax.get_legend_handles_labels()
    # Prints at 6 pt, the same as the median numbers inside Figure 2A
    ax.legend(handles[:1], labels[:1], loc="upper right", frameon=False, handletextpad=0.3,
              fontsize=6 / c.FIG2_PRINT_SCALE)
    c.add_panel_label(ax, "B")
    c.save_figure(fig, "fig02b")


if __name__ == "__main__":
    main()
