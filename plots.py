"""Plotting utilities for XY Monte Carlo simulations."""

import matplotlib.pyplot as plt
import numpy as np

from paths import (
    energy_filename,
    lattice_filename,
    magnetization_filename,
    transitions_filename,
    vortex_count_filename,
    vortex_filename,
)


def default_plot_style() -> dict:
    """Return the default style settings used by simulation plots."""
    return {
        "figsize": (10, 4),
        "linewidth": 1.8,
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "grid_alpha": 0.3,
    }


def make_lattice_figure(model):
    """Create a quiver plot of the current spin configuration."""
    if model.n_dim != 2:
        raise ValueError("Only two-dimensional lattices can be plotted.")

    y_positions, x_positions = np.mgrid[
        0 : model.state.shape[0],
        0 : model.state.shape[1],
    ]

    x_components = np.cos(model.state)
    y_components = np.sin(model.state)

    fig, ax = plt.subplots()

    quiver_plot = ax.quiver(
        x_positions,
        y_positions,
        x_components,
        y_components,
        model.state,
        cmap="twilight",
        norm=plt.Normalize(-np.pi, np.pi),
        scale=40,
        width=0.008,
    )

    fig.colorbar(quiver_plot, ax=ax, label="Angle (radians)")

    ax.set_title("State visualization")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    ax.set_aspect("equal")

    return fig, ax


def save_lattice_plot(model, initial: bool) -> None:
    """Save a lattice plot of the model state."""
    fig, _ = make_lattice_figure(model)
    fig.savefig(lattice_filename(model, initial), bbox_inches="tight")
    plt.close(fig)


def make_magnetization_figure(model, data, style: dict | None = None):
    """Create a plot of the magnetization moving average."""
    if style is None:
        style = default_plot_style()

    if len(data.magnetization_moving_average) == 0:
        raise ValueError("No magnetization data to plot.")

    sweeps = np.arange(
        data.window_size - 1,
        data.window_size - 1 + len(data.magnetization_moving_average),
    )

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(
        sweeps,
        data.magnetization_moving_average,
        linewidth=style["linewidth"],
    )

    ax.set_title("Magnetization modulus of mean vector", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("|<M>|", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_magnetization_plot(model, data, style: dict | None = None) -> None:
    """Save the magnetization moving-average plot."""
    if len(data.magnetization_moving_average) == 0:
        return

    fig, _ = make_magnetization_figure(model, data, style)
    fig.savefig(magnetization_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_transition_figure(model, data, style: dict | None = None):
    """Create a plot of the acceptance ratio per sweep."""
    if style is None:
        style = default_plot_style()

    if len(data.acceptance_ratio) == 0:
        raise ValueError("No transition data to plot.")

    sweeps = np.arange(len(data.acceptance_ratio))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(sweeps, data.acceptance_ratio, linewidth=style["linewidth"])

    ax.set_title("Acceptance ratio per sweep", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("Ratio", fontsize=style["label_size"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_transition_plot(model, data, style: dict | None = None) -> None:
    """Save the transition acceptance-ratio plot."""
    if len(data.acceptance_ratio) == 0:
        return

    fig, _ = make_transition_figure(model, data, style)
    fig.savefig(transitions_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_energy_figure(data, style: dict | None = None):
    """Create a plot of the energy per spin."""
    if style is None:
        style = default_plot_style()

    if len(data.energy_per_spin) == 0:
        raise ValueError("No energy data to plot.")

    sweeps = np.arange(len(data.energy_per_spin))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(sweeps, data.energy_per_spin, linewidth=style["linewidth"])

    ax.set_title("Energy per spin", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("E / N", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_energy_plot(model, data, style: dict | None = None) -> None:
    """Save the energy-per-spin plot."""
    if len(data.energy_per_spin) == 0:
        return

    fig, _ = make_energy_figure(data, style)
    fig.savefig(energy_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_vortex_figure(data, style: dict | None = None):
    """Create a plot of the vortex density."""
    if style is None:
        style = default_plot_style()

    if len(data.vortex_density) == 0:
        raise ValueError("No vortex data to plot.")

    sweeps = np.arange(len(data.vortex_density))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(sweeps, data.vortex_density, linewidth=style["linewidth"])

    ax.set_title("Vortex density vs sweep", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("Vortex density", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_vortex_plot(model, data, style: dict | None = None) -> None:
    """Save the vortex-density plot."""
    if len(data.vortex_density) == 0:
        return

    fig, _ = make_vortex_figure(data, style)
    fig.savefig(vortex_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_vortex_count_figure(data, style: dict | None = None):
    """Create a plot of vortex and antivortex counts."""
    if style is None:
        style = default_plot_style()

    if len(data.n_vortices) == 0:
        raise ValueError("No vortex data to plot.")

    sweeps = np.arange(len(data.n_vortices))

    fig, ax = plt.subplots(figsize=style["figsize"])

    ax.plot(
        sweeps,
        data.n_vortices,
        label="vortices (+1)",
        linewidth=style["linewidth"],
        linestyle="-",
        alpha=0.8,
    )

    ax.plot(
        sweeps,
        data.n_antivortices,
        label="antivortices (-1)",
        linewidth=style["linewidth"],
        linestyle="--",
        alpha=0.8,
    )

    ax.set_title("Vortices and antivortices vs sweep", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("Count", fontsize=style["label_size"])
    ax.legend()
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_vortex_count_plot(model, data, style: dict | None = None) -> None:
    """Save the vortex and antivortex count plot."""
    if len(data.n_vortices) == 0:
        return

    fig, _ = make_vortex_count_figure(data, style)
    fig.savefig(vortex_count_filename(model), bbox_inches="tight")
    plt.close(fig)


def save_stored_plots(model, data) -> None:
    """Save all standard plots for one completed simulation."""
    save_magnetization_plot(model, data)
    save_transition_plot(model, data)
    save_energy_plot(model, data)
    save_vortex_plot(model, data)
    save_vortex_count_plot(model, data)
