"""Generate noisy synthetic datasets from optimized parameter sets."""

import os

import numpy as np
import pandas as pd

from noise_model import apply_composite_noise
from simulate import simulate_variant_response

MODEL_PARAMS = {
    "P_x": 1e-9,
    "P_y": 1e-9,
    "P_z": 1e-9,
    "IPTG": 0.1e-3,
}

SIM_DURATION_S = 420 * 60
SIM_STEP_S = 0.1
SAMPLE_STEP = 6000
VARIANT = "AAV"
PARAMS_DIR = "data/fitting_targets_and_references"
# The noise is not seeded, so every run produces new random datasets. They go here
# and never over the 15 datasets used in the paper, which live in PARAMS_DIR.
OUTPUT_DIR = "results/regenerated_datasets"


def _load_parameters(path: str) -> np.ndarray:
    """Load optimized parameters from a CSV file."""

    df = pd.read_csv(path)
    return df.iloc[0].to_numpy(dtype=float)


def generate_datasets() -> None:
    """Generate and save synthetic datasets.

    Parameters for five different optimizers are loaded from ``PARAMS_DIR``.
    For each optimizer, three independent noisy datasets are created. Each
    dataset contains three replicates. Output files are named
    ``<optimizer>_<index>.csv`` and stored in ``OUTPUT_DIR``.

    Returns:
        None: This function is executed for its side effects only.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    time_s = np.arange(0, SIM_DURATION_S + SIM_STEP_S, SIM_STEP_S)
    sampled_s = time_s[::SAMPLE_STEP]
    time_min = sampled_s / 60

    param_files = {
        "basin_hopping": "basin_hopping_best_params.csv",
        "cmaes": "cmaes_best_params.csv",
        "dual_annealing": "dual_annealing_best_params.csv",
        "lshade": "lshade_best_params.csv",
        "pso": "pso_best_params.csv",
    }

    replicates = 3

    for optimizer, file_name in param_files.items():
        params_path = os.path.join(PARAMS_DIR, file_name)
        params = _load_parameters(params_path)
        clean_trace = simulate_variant_response(
            params=params, model_params=MODEL_PARAMS, variant=VARIANT
        )

        for idx in range(1, 4):
            data = {"time_min": time_min}
            for rep in range(1, replicates + 1):
                noisy = apply_composite_noise(
                    trace=clean_trace, time_array=time_min
                )
                data[f"replicate_{rep}"] = noisy

            df = pd.DataFrame(data)
            path = os.path.join(OUTPUT_DIR, f"{optimizer}_{idx}.csv")
            df.to_csv(path, index=False)


if __name__ == "__main__":
    generate_datasets()
