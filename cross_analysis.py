from pathlib import Path
import csv
from analysis import analyze_simulation
from storage import load_simulation_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_data_folder(data_folder="data"):
    data_folder = Path(data_folder)
    results = []

    for filename in sorted(data_folder.glob("*_data.npz")):
        loaded_data = load_simulation_data(filename)
        result = analyze_simulation(loaded_data)
        result["source_file"] = str(filename)
        results.append(result)

    return results


def save_analysis_summary(results, filename="analysis_summary.csv"):
    if len(results) == 0:
        raise ValueError("No analysis results to save.")

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_with_error_band(
    ax,
    x,
    y,
    yerr=None,
    label=None,
    marker="o",
    linestyle="-",
):
    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        label=label,
    )

    if yerr is not None:
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            alpha=0.15,
        )


def load_summary(filename):
    return pd.read_csv(filename)


def plot_vs_temperature(
    summary,
    observable,
    error=None,
    external_field=None,
    output_folder="analysis_plots",
):
    df = summary.copy()

    if external_field is not None:
        df = df[df["external_field"] == external_field]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for n_particles_1d, group in df.groupby("n_particles_1d"):
        group = group.sort_values("temp")

        x = group["temp"].to_numpy()
        y = group[observable].to_numpy()

        if error is not None:
            yerr = group[error].to_numpy()
        else:
            yerr = None

        plot_with_error_band(
            ax,
            x,
            y,
            yerr=yerr,
            label=f"N={n_particles_1d}",
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
    summary,
    observable,
    error=None,
    temperature=None,
    output_folder="analysis_plots",
):
    df = summary.copy()

    if temperature is not None:
        df = df[df["temp"] == temperature]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for n_particles_1d, group in df.groupby("n_particles_1d"):
        group = group.sort_values("external_field")

        x = group["external_field"].to_numpy()
        y = group[observable].to_numpy()

        if error is not None:
            yerr = group[error].to_numpy()
        else:
            yerr = None

        plot_with_error_band(
            ax,
            x,
            y,
            yerr=yerr,
            label=f"N={n_particles_1d}",
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


def make_standard_plots(
    summary_file,
    output_folder="analysis_plots",
    make_temperature_plots=True,
    make_field_plots=True,
):
    summary = load_summary(summary_file)

    observables = [
        ("mean_absolute_spin", "std_mean_absolute_spin"),
        ("energy_per_spin", "std_energy_per_spin"),
        ("magnetic_susceptibility_per_spin", "std_magnetic_susceptibility_per_spin"),
        ("specific_heat_per_spin", "std_specific_heat_per_spin"),
        ("mean_vortex_density", "std_vortex_density"),
        ("tau", None),
    ]

    for observable, error in observables:
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
    summary,
    x_column,
    y_column,
    error=None,
    output_folder="analysis_plots",
):
    df = summary.copy()

    df = df[(df[x_column] > 0) & (df[y_column] > 0)]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for n_particles_1d, group in df.groupby("n_particles_1d"):
        group = group.sort_values(x_column)

        if error is not None:
            ax.errorbar(
                group[x_column],
                group[y_column],
                yerr=group[error],
                marker="o",
                linestyle="-",
                label=f"N={n_particles_1d}",
            )
        else:
            ax.plot(
                group[x_column],
                group[y_column],
                marker="o",
                linestyle="-",
                label=f"N={n_particles_1d}",
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
    summary,
    observable,
    error=None,
    output_folder="analysis_plots",
):
    df = summary.copy()

    critical_temperature = 0.88

    df["distance_from_tc"] = np.abs(df["temp"] - critical_temperature)

    df = df[(df["distance_from_tc"] > 0) & (df[observable] > 0)]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fit_results = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for n_particles_1d, group in df.groupby("n_particles_1d"):
        group = group.sort_values("temp")

        x = group["distance_from_tc"].to_numpy()
        y = group[observable].to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            linestyle="None",
            label=f"N={n_particles_1d}",
        )

        if len(x) >= 2:
            log_x = np.log(x)
            log_y = np.log(y)

            slope, intercept = np.polyfit(log_x, log_y, 1)

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = np.exp(intercept) * x_fit**slope

            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                label=f"N={n_particles_1d}, alpha={slope:.3f}",
            )

            fit_results.append(
                {
                    "observable": observable,
                    "n_particles_1d": n_particles_1d,
                    "power_law_exponent": slope,
                    "prefactor": np.exp(intercept),
                }
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("|T - Tc|")
    ax.set_ylabel(observable)
    ax.set_title(f"Power-law test: {observable} vs |T - Tc|, Tc={critical_temperature}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()

    filename = output_folder / f"power_law_{observable}_vs_abs_T_minus_Tc.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    return fit_results


def make_temperature_power_law_plots(
    summary_file,
    output_folder="analysis_plots",
    fit_summary_file=None,
):
    summary = load_summary(summary_file)

    observables = [
        ("mean_absolute_spin", "std_mean_absolute_spin"),
        ("energy_per_spin", "std_energy_per_spin"),
        ("magnetic_susceptibility_per_spin", "std_magnetic_susceptibility_per_spin"),
        ("specific_heat_per_spin", "std_specific_heat_per_spin"),
        ("mean_vortex_density", "std_vortex_density"),
        ("tau", None),
    ]

    all_fit_results = []

    for observable, error in observables:
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
    summary,
    observable,
    error=None,
    n_particles_1d=20,
    output_folder="analysis_plots",
):
    df = summary.copy()
    df = df[df["n_particles_1d"] == n_particles_1d]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for temp, group in df.groupby("temp"):
        group = group.sort_values("external_field")

        x = group["external_field"].to_numpy()
        y = group[observable].to_numpy()

        if error is not None:
            yerr = group[error].to_numpy()
        else:
            yerr = None

        plot_with_error_band(
            ax,
            x,
            y,
            yerr=yerr,
            label=f"T={temp:.2f}",
        )

    ax.set_xlabel("External field")
    ax.set_ylabel(observable)
    ax.set_title(f"{observable} vs external field, N={n_particles_1d}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    filename = (
        output_folder
        / f"{observable}_vs_external_field_by_temperature_N_{n_particles_1d}.pdf"
    )
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def make_field_temperature_comparison_plots(
    summary_file,
    output_folder="analysis_plots",
    n_particles_1d=20,
):
    summary = load_summary(summary_file)

    observables = [
        ("mean_absolute_spin", "std_mean_absolute_spin"),
        ("energy_per_spin", "std_energy_per_spin"),
        ("magnetic_susceptibility_per_spin", "std_magnetic_susceptibility_per_spin"),
        ("specific_heat_per_spin", "std_specific_heat_per_spin"),
        ("mean_vortex_density", "std_vortex_density"),
        ("tau", None),
    ]

    for observable, error in observables:
        plot_vs_external_field_by_temperature(
            summary,
            observable=observable,
            error=error,
            n_particles_1d=n_particles_1d,
            output_folder=output_folder,
        )


def plot_KT_fit(
    summary,
    observable,
    Tc=0.88,
    output_folder="analysis_plots",
):
    df = summary.copy()

    df["x"] = np.abs(df["temp"] - Tc)
    df = df[(df["x"] > 0) & (df[observable] > 0)]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for N, group in df.groupby("n_particles_1d"):
        group = group.sort_values("x")

        x = group["x"].to_numpy()
        y = group[observable].to_numpy()

        transformed_x = 1 / np.sqrt(x)
        log_y = np.log(y)

        ax.plot(
            transformed_x,
            log_y,
            marker="o",
            linestyle="None",
            label=f"N={N}",
        )

        if len(x) >= 2:
            slope, intercept = np.polyfit(transformed_x, log_y, 1)

            x_fit = np.linspace(transformed_x.min(), transformed_x.max(), 200)
            y_fit = slope * x_fit + intercept

            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                label=f"N={N}, a={slope:.3f}",
            )

    ax.set_xlabel("1 / sqrt(|T - Tc|)")
    ax.set_ylabel(f"log({observable})")
    ax.set_title(f"KT fit: {observable}, Tc={Tc}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    filename = output_folder / f"KT_fit_{observable}.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
