"""Diagnostic tools for autocorrelation times and block-size estimates."""

from pathlib import Path

import pandas as pd

from analysis import autocorrelation_time
from storage import load_simulation_data


def inspect_folder(folder: str | Path) -> pd.DataFrame:
    """Analyse all simulations in a folder and estimate autocorrelation issues."""
    results = []

    for filename in sorted(Path(folder).glob("*_data.npz")):
        loaded_data = load_simulation_data(filename)

        thermalization_cut = int(loaded_data.get("thermalization_cut", 0))

        magnetization = loaded_data["magnetization_data"][thermalization_cut:]

        tau = autocorrelation_time(magnetization)

        n_samples = len(magnetization)
        suggested_block_size = int(16 * tau)

        problematic = suggested_block_size >= n_samples

        metadata = loaded_data["metadata"].item()

        results.append(
            {
                "file": str(filename),
                "temp": metadata["temp"],
                "n_particles_1d": metadata["n_particles_1d"],
                "external_field": metadata["external_field"],
                "tau": tau,
                "suggested_block_size": suggested_block_size,
                "n_samples": n_samples,
                "problematic": problematic,
            }
        )

    return pd.DataFrame(results)


def print_problematic_simulations(dataframe: pd.DataFrame) -> None:
    """Print simulations whose suggested block size exceeds the sample count."""
    problematic_dataframe = dataframe[dataframe["problematic"]]

    print("\n=== Problematic simulations ===\n")

    if len(problematic_dataframe) == 0:
        print("No problematic tau values found.")
        return

    columns = [
        "temp",
        "n_particles_1d",
        "external_field",
        "tau",
        "suggested_block_size",
        "n_samples",
        "file",
    ]

    print(problematic_dataframe[columns])


def main() -> None:
    """Run autocorrelation diagnostics for all stored simulations."""
    no_field_dataframe = inspect_folder("data")
    field_dataframe = inspect_folder("data_with_field")

    diagnostics_dataframe = pd.concat(
        [no_field_dataframe, field_dataframe],
        ignore_index=True,
    )

    print_problematic_simulations(diagnostics_dataframe)

    diagnostics_dataframe.to_csv(
        "tau_diagnostics.csv",
        index=False,
    )

    print("\nSaved full diagnostics to tau_diagnostics.csv")


if __name__ == "__main__":
    main()
