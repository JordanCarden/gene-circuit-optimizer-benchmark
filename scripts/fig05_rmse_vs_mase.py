"""Figure 5: fit (RMSE) vs parameter recovery (MASE) for each synthetic run (single column).

Runs on an optimizer's own dataset family are excluded (600 of the 750 runs), the
same filter as Figure 4B.

Output: figures/fig05.pdf (and .png)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)
"""

import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()
    df = df[df["optimizer"] != df["category"]].copy()

    fig, ax = c.new_figure()
    for algo in c.ALGORITHMS:
        runs = df[df["optimizer"] == algo]
        ax.scatter(runs["rmse"], runs["mase"], s=6, color=c.COLOR_MAP[algo],
                   alpha=0.5, linewidths=0)
    for algo in c.ALGORITHMS:
        runs = df[df["optimizer"] == algo]
        ax.scatter(runs["rmse"].median(), runs["mase"].median(), marker="D", s=20,
                   color=c.COLOR_MAP[algo], edgecolors="black", linewidths=0.6,
                   label=c.DISPLAY_NAMES[algo], zorder=3)

    c.tick_axis(ax, "x", 0, 300, 50)
    c.tick_axis(ax, "y", 0, 0.7, 0.1)
    ax.set_xlabel("Fit error (RMSE)")
    ax.set_ylabel("Recovery error (MASE)")
    c.align_ylabel(ax)
    legends = _staggered_legend(ax)
    c.save_figure(fig, "fig05", extra_artists=legends)


LEGEND_POSITION = "top"  # "top" (above the box) or "bottom" (below the x-axis label)


def _staggered_legend(ax) -> list:
    """Legend outside the box in two rows, each as wide as the box (so no runs are hidden).

    Each row is split into equal slots with one entry centered in each: Basin Hopping
    and Dual Annealing in the first row, CMA-ES, L-SHADE and PSO in the second.
    """
    entries = dict(zip(*reversed(ax.get_legend_handles_labels())))  # label -> handle
    row_step = 0.10  # row height in axes fraction
    if LEGEND_POSITION == "top":
        rows = [(["CMA-ES", "L-SHADE", "PSO"], 1.07),
                (["Basin Hopping", "Dual Annealing"], 1.07 + row_step)]
        loc = "lower center"
    else:
        fig = ax.figure
        fig.canvas.draw()
        label_bottom = ax.transAxes.inverted().transform(
            (0, ax.xaxis.label.get_window_extent().y0))[1]
        first = label_bottom - 0.04
        rows = [(["Basin Hopping", "Dual Annealing"], first),
                (["CMA-ES", "L-SHADE", "PSO"], first - row_step)]
        loc = "upper center"
    style = dict(frameon=False, handlelength=1.0, handletextpad=0.4, borderaxespad=0,
                 borderpad=0, loc=loc)
    legends = []
    for labels, y in rows:
        for k, label in enumerate(labels):
            slot_center = (k + 0.5) / len(labels)
            leg = ax.legend([entries[label]], [label], bbox_to_anchor=(slot_center, y), **style)
            ax.add_artist(leg)
            legends.append(leg)
    return legends


if __name__ == "__main__":
    main()
