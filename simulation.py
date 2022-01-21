import numpy as np
import matplotlib.pyplot as plt

from sitepercolation import get_random_sequence, get_state_from_sequence
from clusters import label_cluster

sizes = [4, 8, 16, 32, 64, 128]


def has_percolating_cluster(lattice: np.array) -> bool:

    clusters = label_cluster(lattice)

    # Check first and last row
    row_start = clusters[0]
    row_end = clusters[-1]
    vertical_cluster = (
        np.isin(row_start[np.nonzero(row_start)], row_end[np.nonzero(row_end)]).sum()
        > 0
    )

    # Check first and last column
    col_start = clusters[:, 0]
    col_end = clusters[:, -1]
    horizontal_cluster = (
        np.isin(col_start[np.nonzero(col_start)], col_end[np.nonzero(col_end)]).sum()
        > 0
    )

    return vertical_cluster or horizontal_cluster


def percolation_density(sequence: np.array) -> int:
    """Determine de density at which a given sequence percolates"""
    volume = len(sequence)
    for i in range(volume):
        lattice = get_state_from_sequence(sequence, i)
        if has_percolating_cluster(lattice):
            return i / volume


rng = np.random.default_rng(42)


def run_simulation(
    size: int, num_of_simulations: int, rng: np.random.Generator = None, seed: int = 42
) -> np.array:
    if not rng:
        rng = np.random.default_rng(seed=seed)

    densities = np.zeros(num_of_simulations)

    with np.nditer(densities, op_flags=["readwrite"]) as it:
        for simulation in it:
            sequence = get_random_sequence(size, rng)
            simulation[...] = percolation_density(sequence)

    return densities


size = 4
number_of_runs = 1
rng = np.random.default_rng(seed=42)
percolation_densities = run_simulation(size, number_of_runs, rng)

np.savetxt(f"data/run{number_of_runs}_size{size}.txt", percolation_densities)
