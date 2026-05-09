"""Cross-simulation analysis and plotting utilities."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import analyze_simulation
from storage import load_simulation_data

CRITICAL_TEMPERATURE = 0.88


def analysis_plot_style() -> dict:
    """Return default style settings for analysis plots."""
    return {
        "figsize": (4.2, 2.9),
        "linewidth": 1.8,
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "grid_alpha": 0.3,
    }


def analyze_data_folder(data_folder: str | Path = "data") -> list[dict]:
    """Analyse all simulation archives in a folder."""
    data_folder = Path(data_folder)
    results = []

    for filename in sorted(data_folder.glob("*_data.npz")):
        loaded_data = load_simulation_data(filename)
        result = analyze_simulation(loaded_data)
        result["source_file"] = str(filename)
        results.append(result)

    return results


def save_analysis_summary(results: list[dict], filename: str | Path) -> None:
    """Save analysis results to a CSV file."""
    if len(results) == 0:
        raise ValueError("No analysis results to save.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_summary(filename: str | Path) -> pd.DataFrame:
    """Load a CSV summary file."""
    return pd.read_csv(filename)


def plot_with_error_band(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray | None = None,
    label: str | None = None,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    """Plot values and optionally draw an error band."""
    ax.plot(
        x_values,
        y_values,
        marker=marker,
        linestyle=linestyle,
        label=label,
    )

    if y_errors is not None:
        ax.fill_between(
            x_values,
            y_values - y_errors,
            y_values + y_errors,
            alpha=0.15,
        )


def plot_vs_temperature(
    summary: pd.DataFrame,
    observable: str,
    error: str | None = None,
    external_field: float | None = None,
    output_folder: str | Path = "analysis_plots",
    logy: bool = False,
) -> None:
    """Plot one observable as a function of temperature."""
    dataframe = summary.copy()

    if external_field is not None:
        dataframe = dataframe[dataframe["external_field"] == external_field]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    style = analysis_plot_style()
    fig, ax = plt.subplots(figsize=style["figsize"])

    for lattice_size, group in dataframe.groupby("n_particles_1d"):
        group = group.sort_values("temp")

        temperatures = group["temp"].to_numpy()
        values = group[observable].to_numpy()
        errors = group[error].to_numpy() if error is not None else None

        plot_with_error_band(
            ax,
            temperatures,
            values,
            y_errors=errors,
            label=f"N={lattice_size}",
        )

    ax.set_xlabel(
        r"$k_B T / J$",
        fontsize=style["label_size"],
    )

    ax.set_ylabel(
        observable_ylabel(observable),
        fontsize=style["label_size"],
    )

    ax.tick_params(axis="both", labelsize=style["tick_size"])
    if logy:
        ax.set_yscale("log")
    # ax.set_title(f"{observable} vs temperature")
    ax.axvline(
        x=CRITICAL_TEMPERATURE,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=r"$T_c = 0.88$",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if external_field is None:
        suffix = "_logy" if logy else ""
        filename = output_folder / f"{observable}_vs_temperature{suffix}.pdf"
    else:
        suffix = "_logy" if logy else ""
        filename = (
            output_folder
            / f"{observable}_vs_temperature_h_{external_field}{suffix}.pdf"
        )

    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_vs_external_field(
    summary: pd.DataFrame,
    observable: str,
    error: str | None = None,
    temperature: float | None = None,
    output_folder: str | Path = "analysis_plots",
) -> None:
    """Plot one observable as a function of external field."""
    dataframe = summary.copy()

    if temperature is not None:
        dataframe = dataframe[dataframe["temp"] == temperature]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    style = analysis_plot_style()
    fig, ax = plt.subplots(figsize=style["figsize"])

    for lattice_size, group in dataframe.groupby("n_particles_1d"):
        group = group.sort_values("external_field")

        external_fields = group["external_field"].to_numpy()
        values = group[observable].to_numpy()
        errors = group[error].to_numpy() if error is not None else None

        plot_with_error_band(
            ax,
            external_fields,
            values,
            y_errors=errors,
            label=f"N={lattice_size}",
        )

    ax.set_xlabel(
        r"$h/J$",
        fontsize=style["label_size"],
    )

    ax.set_ylabel(
        observable_ylabel(observable),
        fontsize=style["label_size"],
    )

    ax.tick_params(axis="both", labelsize=style["tick_size"])
    # ax.set_title(f"{observable} vs external field")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if temperature is None:
        filename = output_folder / f"{observable}_vs_external_field.pdf"
    else:
        filename = output_folder / f"{observable}_vs_external_field_T_{temperature}.pdf"

    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def standard_observables() -> list[tuple[str, str | None]]:
    """Return the standard observables and their error columns."""
    return [
        ("mean_absolute_spin", "std_mean_absolute_spin"),
        ("energy_per_spin", "std_energy_per_spin"),
        ("magnetic_susceptibility_per_spin", "std_magnetic_susceptibility_per_spin"),
        ("specific_heat_per_spin", "std_specific_heat_per_spin"),
        ("mean_vortex_density", "std_vortex_density"),
        ("tau", None),
    ]


def make_standard_plots(
    summary_file: str | Path,
    output_folder: str | Path = "analysis_plots",
    make_temperature_plots: bool = True,
    make_field_plots: bool = True,
) -> None:
    """Create the standard temperature and field plots."""
    summary = load_summary(summary_file)

    for observable, error in standard_observables():
        if make_temperature_plots:
            plot_vs_temperature(
                summary,
                observable=observable,
                error=error,
                output_folder=output_folder,
            )

        if make_field_plots:
            plot_vs_external_field(
                summary,
                observable=observable,
                error=error,
                output_folder=output_folder,
            )


def plot_vs_external_field_by_temperature(
    summary: pd.DataFrame,
    observable: str,
    error: str | None = None,
    lattice_size: int = 20,
    output_folder: str | Path = "analysis_plots",
    logy: bool = False,
) -> None:
    """Plot an observable against field strength for each temperature."""
    dataframe = summary.copy()
    dataframe = dataframe[dataframe["n_particles_1d"] == lattice_size]

    selected_temperatures = [0.5, 0.7, 0.88, 1.3, 2.5]

    dataframe = dataframe[
        np.any(
            np.isclose(
                dataframe["temp"].to_numpy()[:, None],
                selected_temperatures,
                atol=1e-6,
            ),
            axis=1,
        )
    ]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    style = analysis_plot_style()
    fig, ax = plt.subplots(figsize=style["figsize"])

    for temperature, group in dataframe.groupby("temp"):
        group = group.sort_values("external_field")

        external_fields = group["external_field"].to_numpy()
        values = group[observable].to_numpy()
        errors = group[error].to_numpy() if error is not None else None

        plot_with_error_band(
            ax,
            external_fields,
            values,
            y_errors=errors,
            label=f"T={temperature:.2f}",
        )

    ax.set_xlabel(
        r"$h/J$",
        fontsize=style["label_size"],
    )

    ax.set_ylabel(
        observable_ylabel(observable),
        fontsize=style["label_size"],
    )

    ax.tick_params(axis="both", labelsize=style["tick_size"])
    if logy:
        ax.set_yscale("log")
    # ax.set_title(f"{observable} vs external field, N={lattice_size}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    suffix = "_logy" if logy else ""

    filename = (
        output_folder
        / f"{observable}_vs_external_field_by_temperature_N_{lattice_size}{suffix}.pdf"
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def make_field_temperature_comparison_plots(
    summary_file: str | Path,
    output_folder: str | Path = "analysis_plots",
    lattice_size: int = 20,
) -> None:
    """Create field-dependence plots grouped by temperature."""
    summary = load_summary(summary_file)

    for observable, error in standard_observables():
        plot_vs_external_field_by_temperature(
            summary,
            observable=observable,
            error=error,
            lattice_size=lattice_size,
            output_folder=output_folder,
        )

    plot_vs_external_field_by_temperature(
        summary,
        observable="tau",
        error=None,
        lattice_size=lattice_size,
        output_folder=output_folder,
        logy=True,
    )

    plot_vs_external_field_by_temperature(
        summary,
        observable="magnetic_susceptibility_per_spin",
        error="std_magnetic_susceptibility_per_spin",
        lattice_size=lattice_size,
        output_folder=output_folder,
        logy=True,
    )


def observable_ylabel(observable: str) -> str:
    """Return publication-quality y-axis labels."""

    labels = {
        "energy_per_spin": r"$e/J$",
        "mean_absolute_spin": r"$\langle |m| \rangle$",
        "magnetic_susceptibility_per_spin": r"$J \chi_m$",
        "specific_heat_per_spin": r"$C/k_B$",
        "mean_vortex_density": r"$\rho_v$",
        "tau": r"$\tau$",
    }

    return labels.get(observable, observable)
