import numpy as np
import matplotlib.pyplot as plt
from paths import lattice_filename, magnetization_filename, transitions_filename


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
        model.n_particles,
        model.n_particles + len(data.magnetization_moving_average),
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

    x = np.arange(
        model.n_particles,
        model.n_particles + len(data.acceptance_ratio),
    )

    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.plot(x, data.acceptance_ratio, linewidth=style["linewidth"])

    ax.set_title(
        f"Acceptance ratio in the last {model.n_particles} iterations",
        fontsize=style["title_size"],
    )
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


def save_stored_plots(model, data):
    save_magnetization_plot(model, data)
    save_transition_plot(model, data)
