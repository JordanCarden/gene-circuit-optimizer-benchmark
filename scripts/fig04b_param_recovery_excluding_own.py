"""Figure 4B: parameter recovery (MASE) with each optimizer's own dataset family removed.

Output: figures/fig04b.pdf (and .png)
Data:   data/synthetic_results.csv
        data/fitting_targets_and_references/<family>_best_params.csv (ground truth)
"""

import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()
    df = df[df["optimizer"] != df["category"]].copy()

    medians = df.groupby("algorithm")["mase"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = c.new_figure()
    c.violin_plot(ax, df, "algorithm", "mase", order, c.DISPLAY_COLOR_MAP,
                   median_fmt=".3f")

    ax.set_ylabel("MASE")
    c.align_ylabel(ax)
    c.style_optimizer_xticks(ax, order)

    ax.set_xlim(-0.5, len(order) - 0.5)
    c.tick_axis(ax, "y", 0, 0.7, 0.1)
    c.add_panel_label(ax, "B")
    c.save_figure(fig, "fig04b")


if __name__ == "__main__":
    main()
