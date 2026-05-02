import numpy as np
import matplotlib.pyplot as plt
from paths import (
    lattice_filename,
    magnetization_filename,
    transitions_filename,
    energy_filename,
    vortex_filename,
    vortex_count_filename,
)


def default_plot_style():
    return {
        "figsize": (10, 4),
        "linewidth": 1.8,
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "grid_alpha": 0.3,
    }


def make_lattice_figure(model):
    if model.n_dim != 2:
        raise ValueError("Only 2d plot")

    y, x = np.mgrid[0 : model.state.shape[0], 0 : model.state.shape[1]]

    u = np.cos(model.state)
    v = np.sin(model.state)

    fig, ax = plt.subplots()

    q = ax.quiver(
        x,
        y,
        u,
        v,
        model.state,
        cmap="twilight",
        norm=plt.Normalize(-np.pi, np.pi),
        scale=40,
        width=0.008,
    )

    fig.colorbar(q, ax=ax, label="Angle (radians)")

    ax.set_title("State visualization")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    ax.set_aspect("equal")

    return fig, ax


def save_lattice_plot(model, initial: bool):
    fig, _ = make_lattice_figure(model)

    fig.savefig(lattice_filename(model, initial), bbox_inches="tight")
    plt.close(fig)


def make_magnetization_figure(model, data, style=None):
    if style is None:
        style = default_plot_style()

    if len(data.magnetization_moving_average) == 0:
        raise ValueError("No magnetization data to plot.")

    x = np.arange(
        data.window_size - 1,
        data.window_size - 1 + len(data.magnetization_moving_average),
    )

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(x, data.magnetization_moving_average, linewidth=style["linewidth"])

    ax.set_title("Magnetization modulus of mean vector", fontsize=style["title_size"])
    ax.set_xlabel("Iteration", fontsize=style["label_size"])
    ax.set_ylabel("|<M>|", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_magnetization_plot(model, data, style=None):
    if len(data.magnetization_moving_average) == 0:
        return

    fig, _ = make_magnetization_figure(model, data, style)
    fig.savefig(magnetization_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_transition_figure(model, data, style=None):
    if style is None:
        style = default_plot_style()

    if len(data.acceptance_ratio) == 0:
        raise ValueError("No transition data to plot.")

    x = np.arange(len(data.acceptance_ratio))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(x, data.acceptance_ratio, linewidth=style["linewidth"])

    ax.set_title("Acceptance ratio per sweep", fontsize=style["title_size"])
    ax.set_xlabel("Iteration", fontsize=style["label_size"])
    ax.set_ylabel("Ratio", fontsize=style["label_size"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_transition_plot(model, data, style=None):
    if len(data.acceptance_ratio) == 0:
        return

    fig, _ = make_transition_figure(model, data, style)
    fig.savefig(transitions_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_energy_figure(data, style=None):
    if style is None:
        style = default_plot_style()

    if len(data.energy_per_spin) == 0:
        raise ValueError("No energy data to plot.")

    x = np.arange(len(data.energy_per_spin))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(x, data.energy_per_spin, linewidth=style["linewidth"])

    ax.set_title("Energy per spin", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("E / N", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_energy_plot(model, data, style=None):
    if len(data.energy_per_spin) == 0:
        return

    fig, _ = make_energy_figure(data, style)
    fig.savefig(energy_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_vortex_figure(data, style=None):
    if style is None:
        style = default_plot_style()

    if len(data.vortex_density) == 0:
        raise ValueError("No vortex data to plot.")

    x = np.arange(len(data.vortex_density))

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(x, data.vortex_density, linewidth=style["linewidth"])

    ax.set_title("Vortex density vs sweep", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("Vortex density", fontsize=style["label_size"])
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_vortex_plot(model, data, style=None):
    if len(data.vortex_density) == 0:
        return

    fig, _ = make_vortex_figure(data, style)
    fig.savefig(vortex_filename(model), bbox_inches="tight")
    plt.close(fig)


def make_vortex_count_figure(data, style=None):
    if style is None:
        style = default_plot_style()

    if len(data.n_vortices) == 0:
        raise ValueError("No vortex data to plot.")

    x = np.arange(len(data.n_vortices))

    fig, ax = plt.subplots(figsize=style["figsize"])

    ax.plot(x, data.n_vortices, label="vortices (+1)", linewidth=style["linewidth"])
    ax.plot(
        x,
        data.n_antivortices,
        label="antivortices (-1)",
        linewidth=style["linewidth"],
    )

    ax.set_title("Vortices and antivortices vs sweep", fontsize=style["title_size"])
    ax.set_xlabel("Sweep", fontsize=style["label_size"])
    ax.set_ylabel("Count", fontsize=style["label_size"])

    ax.legend()
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])

    fig.tight_layout()
    return fig, ax


def save_vortex_count_plot(model, data, style=None):
    if len(data.n_vortices) == 0:
        return

    fig, _ = make_vortex_count_figure(data, style)
    fig.savefig(vortex_count_filename(model), bbox_inches="tight")
    plt.close(fig)


def save_stored_plots(model, data):
    save_magnetization_plot(model, data)
    save_transition_plot(model, data)
    save_energy_plot(model, data)
    save_vortex_plot(model, data)
    save_vortex_count_plot(model, data)
