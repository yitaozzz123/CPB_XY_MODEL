from main import XY_Monte_Carlo
import numpy as np


def generate_data(
    tau,
    temp,
    n_particles_1d,
    external_field=0,
    all_up=False,
    seed=None,
    n_iterations=1000,
):
    model = XY_Monte_Carlo(
        temp,
        n_particles_1d,
        external_field=0,
        all_up=False,
        seed=None,
        n_iterations=1000,
    )

    model.full_transition()
    magnetization = np.array(model.magnetization_data)
    return magnetization, tau


class thermal_analysis:
    def __init__(
        self,
        tau,
        model,
    ):
        self.model = model
        self.tau = tau
