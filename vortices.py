import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def vortex_map(state):
    theta = state

    theta_right = np.roll(theta, shift=-1, axis=1)
    theta_down = np.roll(theta, shift=-1, axis=0)
    theta_down_right = np.roll(theta_down, shift=-1, axis=1)

    total_angle = (
        wrap_angle(theta_down - theta)
        + wrap_angle(theta_down_right - theta_down)
        + wrap_angle(theta_right - theta_down_right)
        + wrap_angle(theta - theta_right)
    )

    return np.rint(total_angle / (2 * np.pi)).astype(int)


def count_vortices(state):
    vortices = vortex_map(state)

    n_vortices = np.sum(vortices == 1)
    n_antivortices = np.sum(vortices == -1)

    return n_vortices, n_antivortices


def vortex_density(state):
    n_vortices, n_antivortices = count_vortices(state)
    n_plaquettes = state.shape[0] * state.shape[1]

    return (n_vortices + n_antivortices) / n_plaquettes
