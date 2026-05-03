from pathlib import Path
import csv
from analysis import analyze_simulation
from storage import load_simulation_data
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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

        if error is not None:
            ax.errorbar(
                group["temp"],
                group[observable],
                yerr=group[error],
                marker="o",
                linestyle="-",
                label=f"N={n_particles_1d}",
            )
        else:
            ax.plot(
                group["temp"],
                group[observable],
                marker="o",
                linestyle="-",
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

        if error is not None:
            ax.errorbar(
                group["external_field"],
                group[observable],
                yerr=group[error],
                marker="o",
                linestyle="-",
                label=f"N={n_particles_1d}",
            )
        else:
            ax.plot(
                group["external_field"],
                group[observable],
                marker="o",
                linestyle="-",
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


def make_standard_plots(summary_file, output_folder="analysis_plots"):
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
        plot_vs_temperature(
            summary,
            observable=observable,
            error=error,
            output_folder=output_folder,
        )

        plot_vs_external_field(
            summary,
            observable=observable,
            error=error,
            output_folder=output_folder,
        )
