"""Figure 2A: best RMSE of each of the 13 runs per optimizer on the experimental data.

Output: figures/fig02a.pdf (and .png)
Data:   data/experimental_results/*_history_variant_AAV_run_*.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as c


def main() -> None:
    c.apply_publication_style(text_scale=1 / c.FIG2_PRINT_SCALE)
    num_points = len(c.load_experimental_trace())
    rows = [
        {"algorithm": c.DISPLAY_NAMES[algo], "rmse": np.sqrt(hist["sse"].min() / num_points)}
        for algo, _, hist in c.iter_run_histories()
    ]
    df = pd.DataFrame(rows)

    medians = df.groupby("algorithm")["rmse"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = c.new_figure()
    c.violin_plot(ax, df, "algorithm", "rmse", order, c.DISPLAY_COLOR_MAP,
                   median_fmt=".2f", median_size=6)  # smaller: 2A prints narrower

    ax.set_ylabel("RMSE (a.u.)")
    c.align_ylabel(ax)
    c.style_optimizer_xticks(ax, order)

    ax.set_xlim(-0.5, len(order) - 0.5)
    c.tick_axis(ax, "y", 0, 350, 50)  # same range as Figure 3
    c.add_panel_label(ax, "A")
    c.save_figure(fig, "fig02a")


if __name__ == "__main__":
    main()
