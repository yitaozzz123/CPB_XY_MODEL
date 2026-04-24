from main import XY_Monte_Carlo
import numpy as np


def transition(
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
