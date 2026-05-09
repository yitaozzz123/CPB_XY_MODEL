from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import autocorrelation_time
from storage import load_simulation_data
from matplotlib.colors import LogNorm
from cross_analysis import analysis_plot_style


def compute_tau_from_file(filename: str | Path) -> float:
    """Compute tau from one saved simulation archive."""

    loaded_data = load_simulation_data(filename)

    thermalization_cut = int(loaded_data.get("thermalization_cut", 0))
    magnetization = loaded_data["magnetization_data"][thermalization_cut:]

    if len(magnetization) == 0:
        raise ValueError(
            f"No magnetization data after thermalization cut in {filename}"
        )

    return float(autocorrelation_time(magnetization))


def read_temp_and_field_from_file(filename: str | Path) -> tuple[float, float]:
    """Read temperature and external field from saved simulation archive."""

    loaded_data = load_simulation_data(filename)
    metadata = loaded_data["metadata"].item()

    temperature = float(metadata["temp"])
    field = float(metadata["external_field"])

    return temperature, field


def plot_tau_phase_diagram(
    tau_function,
    data_roots: list[Path] | tuple[Path, ...] = (
        Path("data/tau_study"),
        Path("data_with_field/tau_study"),
    ),
) -> None:
    """Create a scatter plot of tau(T, h)."""

    temperatures = []
    fields = []
    taus = []

    for data_root in data_roots:
        data_root = Path(data_root)

        if not data_root.exists():
            print(f"Skipping missing folder: {data_root}")
            continue

        for filepath in sorted(data_root.rglob("*_data.npz")):
            try:
                temperature, field = read_temp_and_field_from_file(filepath)
                tau = tau_function(filepath)

                temperatures.append(temperature)
                fields.append(field)
                taus.append(tau)

                print(f"Loaded T={temperature:.2f}, h={field:.2f}, tau={tau:.3f}")

            except Exception as exc:
                print(f"Skipping {filepath}: {exc}")

    if len(taus) == 0:
        raise ValueError(f"No valid data files found in {data_roots}")

    temperatures = np.array(temperatures)
    fields = np.array(fields)
    taus = np.array(taus)

    style = analysis_plot_style()

    fig, ax = plt.subplots(figsize=style["figsize"])

    sc = plt.scatter(
        fields,
        temperatures,
        c=taus,
        s=80,
        cmap="inferno",
        norm=LogNorm(),
        edgecolors="black",
        linewidths=0.4,
    )

    plt.colorbar(sc, label=r"$\tau$")
    plt.xlabel(r"$h/J$")
    plt.ylabel(r"$k_B T/J$")
    plt.title(r"Relaxation time $\tau(T, h)$")

    plt.tight_layout()
    plt.show()


def analysis_temp_vs_field():
    plot_tau_phase_diagram(
        tau_function=compute_tau_from_file,
        data_roots=[
            Path("data/tau_study"),
            Path("data_with_field/tau_study"),
        ],
    )


def plot_tau_vs_temperature_for_fields(
    selected_fields=(0.0, 0.25, 0.5),
    data_roots=(Path("data_with_field"),),
    output_folder="analysis_with_field/temperature_comparison_plots",
):
    """Plot tau vs temperature for selected external fields."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    field_data = {field: {"T": [], "tau": []} for field in selected_fields}

    for data_root in data_roots:
        data_root = Path(data_root)

        if not data_root.exists():
            continue

        for filepath in sorted(data_root.rglob("*_data.npz")):
            try:
                temperature, field = read_temp_and_field_from_file(filepath)

                matched_field = None
                for selected_field in selected_fields:
                    if np.isclose(field, selected_field, atol=1e-6):
                        matched_field = selected_field
                        break

                if matched_field is None:
                    continue

                tau = compute_tau_from_file(filepath)

                field_data[matched_field]["T"].append(temperature)
                field_data[matched_field]["tau"].append(tau)

            except Exception as exc:
                print(f"Skipping {filepath}: {exc}")

    style = analysis_plot_style()

    fig, ax = plt.subplots(figsize=style["figsize"])

    for field in selected_fields:
        temperatures = np.array(field_data[field]["T"])
        taus = np.array(field_data[field]["tau"])

        order = np.argsort(temperatures)

        ax.plot(
            temperatures[order],
            taus[order],
            marker="o",
            linewidth=style["linewidth"],
            label=rf"$h/J = {field}$",
        )

    ax.set_xlabel(r"$k_B T / J$", fontsize=style["label_size"])
    ax.set_ylabel(r"$\tau$", fontsize=style["label_size"])

    ax.set_yscale("log")

    ax.grid(True, alpha=style["grid_alpha"])

    ax.tick_params(axis="both", labelsize=style["tick_size"])

    ax.legend()

    fig.tight_layout()

    filename = output_folder / "tau_vs_temperature_selected_fields.pdf"

    fig.savefig(filename, bbox_inches="tight")

    plt.close(fig)
