"""Figure 3: fitting accuracy (RMSE) of each optimizer over all 750 synthetic runs.

Output: figures/fig03.pdf (and .png)
Data:   data/synthetic_results.csv
"""

import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style()
    df = c.load_synthetic_results()

    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = c.new_figure()
    c.violin_plot(ax, df, "algorithm", "rmse", order, c.DISPLAY_COLOR_MAP,
                   median_fmt=".2f")

    ax.set_ylabel("RMSE (a.u.)")
    c.align_ylabel(ax)
    c.style_optimizer_xticks(ax, order)

    ax.set_xlim(-0.5, len(order) - 0.5)
    c.tick_axis(ax, "y", 0, 350, 50)
    c.save_figure(fig, "fig03")


if __name__ == "__main__":
    main()
