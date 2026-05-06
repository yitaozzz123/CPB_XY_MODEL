"""Cross-simulation analysis and plotting utilities."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import analyze_simulation
from storage import load_simulation_data

CRITICAL_TEMPERATURE = 0.88


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
) -> None:
    """Plot one observable as a function of temperature."""
    dataframe = summary.copy()

    if external_field is not None:
        dataframe = dataframe[dataframe["external_field"] == external_field]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

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

    ax.set_xlabel("Temperature")
    ax.set_ylabel(observable)
    ax.set_title(f"{observable} vs temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if external_field is None:
        filename = output_folder / f"{observable}_vs_temperature.pdf"
    else:
        filename = output_folder / f"{observable}_vs_temperature_h_{external_field}.pdf"

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

    fig, ax = plt.subplots(figsize=(8, 5))

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

    ax.set_xlabel("External field")
    ax.set_ylabel(observable)
    ax.set_title(f"{observable} vs external field")
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


def plot_loglog(
    summary: pd.DataFrame,
    x_column: str,
    y_column: str,
    error: str | None = None,
    output_folder: str | Path = "analysis_plots",
) -> None:
    """Create a log-log plot of one observable against another."""
    dataframe = summary.copy()
    dataframe = dataframe[(dataframe[x_column] > 0) & (dataframe[y_column] > 0)]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for lattice_size, group in dataframe.groupby("n_particles_1d"):
        group = group.sort_values(x_column)

        if error is not None:
            ax.errorbar(
                group[x_column],
                group[y_column],
                yerr=group[error],
                marker="o",
                linestyle="-",
                label=f"N={lattice_size}",
            )
        else:
            ax.plot(
                group[x_column],
                group[y_column],
                marker="o",
                linestyle="-",
                label=f"N={lattice_size}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"log-log plot: {y_column} vs {x_column}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = output_folder / f"loglog_{y_column}_vs_{x_column}.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_power_law(
    summary: pd.DataFrame,
    observable: str,
    error: str | None = None,
    output_folder: str | Path = "analysis_plots",
) -> list[dict]:
    """Plot a power-law test against distance from the critical temperature."""
    dataframe = summary.copy()

    dataframe["distance_from_tc"] = np.abs(dataframe["temp"] - CRITICAL_TEMPERATURE)
    dataframe = dataframe[
        (dataframe["distance_from_tc"] > 0) & (dataframe[observable] > 0)
    ]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fit_results = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for lattice_size, group in dataframe.groupby("n_particles_1d"):
        group = group.sort_values("temp")

        distances = group["distance_from_tc"].to_numpy()
        values = group[observable].to_numpy()

        ax.plot(
            distances,
            values,
            marker="o",
            linestyle="None",
            label=f"N={lattice_size}",
        )

        if len(distances) >= 2:
            log_distances = np.log(distances)
            log_values = np.log(values)

            slope, intercept = np.polyfit(log_distances, log_values, 1)

            distances_fit = np.linspace(distances.min(), distances.max(), 200)
            values_fit = np.exp(intercept) * distances_fit**slope

            ax.plot(
                distances_fit,
                values_fit,
                linestyle="--",
                label=f"N={lattice_size}, alpha={slope:.3f}",
            )

            fit_results.append(
                {
                    "observable": observable,
                    "n_particles_1d": lattice_size,
                    "power_law_exponent": slope,
                    "prefactor": np.exp(intercept),
                }
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|T - Tc|")
    ax.set_ylabel(observable)
    ax.set_title(f"Power-law test: {observable} vs |T - Tc|, Tc={CRITICAL_TEMPERATURE}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()

    filename = output_folder / f"power_law_{observable}_vs_abs_T_minus_Tc.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    return fit_results


def make_temperature_power_law_plots(
    summary_file: str | Path,
    output_folder: str | Path = "analysis_plots",
    fit_summary_file: str | Path | None = None,
) -> list[dict]:
    """Create power-law plots for all standard observables."""
    summary = load_summary(summary_file)
    all_fit_results = []

    for observable, error in standard_observables():
        fit_results = plot_temperature_power_law(
            summary,
            observable=observable,
            error=error,
            output_folder=output_folder,
        )
        all_fit_results.extend(fit_results)

    if fit_summary_file is not None:
        fit_summary_file = Path(fit_summary_file)
        fit_summary_file.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(all_fit_results).to_csv(fit_summary_file, index=False)

    return all_fit_results


def plot_vs_external_field_by_temperature(
    summary: pd.DataFrame,
    observable: str,
    error: str | None = None,
    lattice_size: int = 20,
    output_folder: str | Path = "analysis_plots",
) -> None:
    """Plot an observable against field strength for each temperature."""
    dataframe = summary.copy()
    dataframe = dataframe[dataframe["n_particles_1d"] == lattice_size]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

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

    ax.set_xlabel("External field")
    ax.set_ylabel(observable)
    ax.set_title(f"{observable} vs external field, N={lattice_size}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    filename = (
        output_folder
        / f"{observable}_vs_external_field_by_temperature_N_{lattice_size}.pdf"
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


def plot_KT_fit(
    summary: pd.DataFrame,
    observable: str,
    critical_temperature: float = CRITICAL_TEMPERATURE,
    output_folder: str | Path = "analysis_plots",
) -> None:
    """Create a Kosterlitz-Thouless fit diagnostic plot."""
    dataframe = summary.copy()

    dataframe["distance_from_tc"] = np.abs(dataframe["temp"] - critical_temperature)
    dataframe = dataframe[
        (dataframe["distance_from_tc"] > 0) & (dataframe[observable] > 0)
    ]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for lattice_size, group in dataframe.groupby("n_particles_1d"):
        group = group.sort_values("distance_from_tc")

        distances = group["distance_from_tc"].to_numpy()
        values = group[observable].to_numpy()

        transformed_distances = 1 / np.sqrt(distances)
        log_values = np.log(values)

        ax.plot(
            transformed_distances,
            log_values,
            marker="o",
            linestyle="None",
            label=f"N={lattice_size}",
        )

        if len(distances) >= 2:
            slope, intercept = np.polyfit(transformed_distances, log_values, 1)

            fit_distances = np.linspace(
                transformed_distances.min(),
                transformed_distances.max(),
                200,
            )
            fit_values = slope * fit_distances + intercept

            ax.plot(
                fit_distances,
                fit_values,
                linestyle="--",
                label=f"N={lattice_size}, a={slope:.3f}",
            )

    ax.set_xlabel("1 / sqrt(|T - Tc|)")
    ax.set_ylabel(f"log({observable})")
    ax.set_title(f"KT fit: {observable}, Tc={critical_temperature}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename = output_folder / f"KT_fit_{observable}.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
