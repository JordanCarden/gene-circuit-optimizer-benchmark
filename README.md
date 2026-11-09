# Gene circuit optimizer benchmark

Code and data for *A Comparative Study on Different Optimization Solvers for Mechanistic ODE-based Gene Circuit Model Parameterization* (Carden, Tang, An).

The study compares five global optimizers (CMA-ES, L-SHADE, PSO, Dual Annealing and Basin-Hopping) at fitting a mechanistic ODE model of an I1-FFL gene circuit with 11 states and 21 kinetic parameters. It uses measured GFP data and 15 synthetic datasets with known parameters.

## Setup

Python 3.12. Run every command from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the figures

```bash
python make_all_figures.py           # all figures
python make_all_figures.py fig02b    # one figure
```

Figures are written to `figures/` as vector PDFs with 600 dpi PNG copies.

| Figure | Script | Shows |
|---|---|---|
| 2A | `scripts/fig02a_rmse_experimental.py` | Best RMSE of each of the 13 runs per optimizer on the experimental data |
| 2B | `scripts/fig02b_best_fits.py` | Best-fit GFP curve per optimizer against the measured GFP |
| 2C | `scripts/fig02c_convergence.py` | Best-so-far RMSE against function evaluations (median and IQR) |
| 2 key | `scripts/fig02_legend.py` | Optimizer color key for Figure 2 |
| 3 | `scripts/fig03_rmse_synthetic.py` | RMSE over all 750 synthetic runs |
| 4A | `scripts/fig04a_param_recovery.py` | Median parameter recovery (MASE) for each optimizer on each dataset family |
| 4B | `scripts/fig04b_param_recovery_excluding_own.py` | MASE with each optimizer's own dataset family excluded |
| 5 | `scripts/fig05_rmse_vs_mase.py` | RMSE against MASE per run, excluding own dataset family (600 runs) |

Figure 1 is a circuit diagram with the model equations, so it has no script here.

## Rerun the optimizations

Each run makes about 21,000 ODE-solving objective evaluations, and the experimental fits took a median of about 22 minutes each on our cluster. Run histories are written to `results/`.

Fit the measured data (5 optimizers × 13 seeds = 65 runs):

```bash
for optimizer in cmaes lshade pso dual_annealing basin_hopping; do
  for seed in $(seq 1 13); do
    python python/fit_experimental.py --optimizer "$optimizer" --seed "$seed"
  done
done
```

Fit the synthetic datasets (5 optimizers × 15 datasets × 10 seeds = 750 runs):

```bash
for optimizer in cmaes lshade pso dual_annealing basin_hopping; do
  for family in cmaes lshade pso dual_annealing basin_hopping; do
    for index in 1 2 3; do
      for seed in $(seq 1 10); do
        python python/fit_synthetic.py --optimizer "$optimizer" --dataset "${family}_${index}" --seed "$seed"
      done
    done
  done
done
```

## Code

| File | Purpose |
|---|---|
| `python/protein_model.py` | ODE right-hand side |
| `python/simulate.py` | Simulates GFP for the AAV degradation-tag variant |
| `python/objective_function.py` | SSE objective, with a per-evaluation history |
| `python/optimizers.py` | Settings and wrappers for the five optimizers |
| `python/fit_experimental.py` | Fits the measured GFP trace |
| `python/fit_synthetic.py` | Fits one synthetic dataset |
| `python/noise_model.py`, `python/generate_data.py` | Composite noise model and synthetic dataset generation |
| `scripts/common.py` | Shared figure style, data loading, and the metrics RMSE and MASE |

## Data

| Path | Contents |
|---|---|
| `data/fitting_targets_and_references/experimental_data.mat` | Measured GFP for the AAV variant: 43 points, 0–420 min, every 10 min (key `AAV`) |
| `data/fitting_targets_and_references/<optimizer>_best_params.csv` | Best experimental fit from each optimizer (21 parameters), which is also the ground truth for that synthetic dataset family |
| `data/fitting_targets_and_references/<optimizer>_<1-3>.csv` | The 15 synthetic datasets, each with 3 noisy replicates |
| `data/experimental_results/` | 65 run histories on the experimental data. Each row is one evaluation: `params`, `sse`, `elapsed_time_s` |
| `data/synthetic_results.csv` | Best result of each of the 750 synthetic runs: `optimizer`, `dataset`, `seed`, `min_sse`, `params` |

RMSE = √(SSE / 43). MASE scales each of the 21 parameters to [0, 1] using its search bounds, then averages the absolute error against the ground truth.

## Notes

- The 15 datasets in `data/` are the exact ones used in the paper. The noise in `python/generate_data.py` is not seeded, so running it creates new random datasets. It writes them to `results/regenerated_datasets/` and leaves the originals alone.
- Per-evaluation histories for the 750 synthetic runs are not included because of their size (about 6 GB). `data/synthetic_results.csv` holds the best result of every run, which is all the figures need.
- Keep MealPy at the pinned version. The code uses the `L_SHADE` class from MealPy 3.0.2, and later releases changed the L-SHADE classes.

## License

The code is released under the [MIT License](LICENSE).
