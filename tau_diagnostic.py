from pathlib import Path
import pandas as pd

from storage import load_simulation_data
from analysis import autocorrelation_time


def inspect_folder(folder):
    results = []

    for filename in sorted(Path(folder).glob("*_data.npz")):

        loaded = load_simulation_data(filename)

        thermalization_cut = int(loaded.get("thermalization_cut", 0))

        magnetization = loaded["magnetization_data"][thermalization_cut:]

        tau = autocorrelation_time(magnetization)

        n_samples = len(magnetization)

        suggested_block_size = int(16 * tau)

        problematic = suggested_block_size >= n_samples

        metadata = loaded["metadata"].item()

        results.append(
            {
                "file": str(filename),
                "temp": metadata["temp"],
                "N": metadata["n_particles_1d"],
                "external_field": metadata["external_field"],
                "tau": tau,
                "16_tau": suggested_block_size,
                "n_samples": n_samples,
                "problematic": problematic,
            }
        )

    return pd.DataFrame(results)


df1 = inspect_folder("data")
df2 = inspect_folder("data_with_field")

df = pd.concat([df1, df2], ignore_index=True)

problematic_df = df[df["problematic"]]

print("\n=== Problematic simulations ===\n")

if len(problematic_df) == 0:
    print("No problematic tau values found.")
else:
    print(
        problematic_df[
            [
                "temp",
                "N",
                "external_field",
                "tau",
                "16_tau",
                "n_samples",
                "file",
            ]
        ]
    )

df.to_csv("tau_diagnostics.csv", index=False)

print("\nSaved full diagnostics to tau_diagnostics.csv")
