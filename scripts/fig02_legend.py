"""Figure 2 color key: one legend with every optimizer's color, shared by all panels.

Output: figures/fig02_legend.pdf (and .png)
Placed centered below the three Figure 2 panels, at natural size; its text prints at 7 pt.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import common as c

# Same order as the panels: best to worst median RMSE on the experimental data (2A)
ORDER = ["cmaes", "lshade", "basin_hopping", "dual_annealing", "pso"]


def main() -> None:
    c.apply_publication_style()
    fig = plt.figure(figsize=(7, 0.5))
    # Filled squares with a black outline, one per optimizer
    handles = [Patch(facecolor=c.COLOR_MAP[algo], edgecolor="black", linewidth=0.6,
                     label=c.DISPLAY_NAMES[algo])
               for algo in ORDER]
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               handlelength=0.9, handleheight=0.9, handletextpad=0.5, columnspacing=1.6)
    # Make each swatch exactly square (its height is set by the text line, width by
    # handlelength): shrink the width to match the height, keeping it centered
    fig.canvas.draw()
    for patch in fig.legends[0].get_patches():
        x, y, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
        patch.set_x(x + (w - h) / 2)
        patch.set_width(h)
    c.save_figure(fig, "fig02_legend")


if __name__ == "__main__":
    main()
