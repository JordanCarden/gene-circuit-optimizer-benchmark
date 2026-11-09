"""Figure 4A: median parameter recovery (MASE) for every optimizer x dataset family.

Rows are optimizers, columns are the synthetic dataset family (the optimizer whose
experimental best fit generated the data, i.e. the ground truth). Numbers are the
median MASE of the 30 runs in each cell (3 datasets x 10 seeds). Each cell is shaded
by its rank within the column: the best (lowest) median is dark green with bold white
text, then each lower rank is lighter, down to white for the worst.

Output: figures/fig04a.pdf (and .png)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)
"""

import matplotlib.pyplot as plt
import numpy as np

import common as c

# Rank within the column -> fill (ColorBrewer Greens); 1 = lowest median MASE
RANK_COLORS = {1: "#1E8C45", 2: "#74C476", 3: "#BAE4B3", 4: "#EDF8E9", 5: "#FFFFFF"}


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()

    table = df.pivot_table(index="optimizer", columns="category", values="mase", aggfunc="median")
    table = table.loc[c.ALGORITHMS, c.ALGORITHMS]
    rank = table.rank(axis=0).astype(int)  # full precision, so there are no ties
    labels = [c.DISPLAY_NAMES[a] for a in c.ALGORITHMS]
    n = len(labels)

    fig, ax = c.new_figure()  # same plotting area as every other figure
    for i, opt in enumerate(c.ALGORITHMS):
        for j, fam in enumerate(c.ALGORITHMS):
            r = rank.loc[opt, fam]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=RANK_COLORS[r],
                                       edgecolor="white", linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, f"{table.loc[opt, fam]:.3f}", ha="center", va="center",
                    color="white" if r == 1 else "0.1",
                    fontweight="bold" if r == 1 else "normal")

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xlabel("Synthetic dataset family")
    ax.set_ylabel("Optimizer")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels)
    ax.tick_params(length=0)
    c.add_panel_label(ax, "A")
    c.save_figure(fig, "fig04a")


if __name__ == "__main__":
    main()
