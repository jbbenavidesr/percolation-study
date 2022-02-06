import numpy as np
import matplotlib.pyplot as plt

from percolation.site import SitePercolation
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

    # center = clusters[int(len(col_start) / 2), int(len(row_start) / 2)]
    # print(center)

    # return (
    #     center in row_start
    #     or center in row_end
    #     or center in col_start
    #     or center in col_end
    # )

    return vertical_cluster and horizontal_cluster


def percolation_density(sequence: np.array, borders: bool = True) -> int:
    """Determine de density at which a given sequence percolates"""
    volume = len(sequence)

    i = int(volume / 2)
    step = int(i / 2)
    while True:
        lattice = SitePercolation.state_of_lattice(sequence, i)
        if not borders:
            lattice = lattice[1:-1, 1:-1]

        if has_percolating_cluster(lattice):
            if step == 0:
                i = i - 1
                continue

            i = i - step
            step = int(step / 2)
        else:
            if step == 0:
                if not borders:
                    volume = volume - 4 * (np.sqrt(volume) - 1)
                return (lattice.sum() + 1) / volume
            i = i + step
            step = int(step / 2)


rng = np.random.default_rng(42)


def run_simulation(
    size: int, num_of_simulations: int, rng: np.random.Generator = None, seed: int = 42
) -> np.array:
    perc = SitePercolation(size, rng, seed)

    densities = np.zeros(num_of_simulations)

    with np.nditer(densities, op_flags=["readwrite"]) as it:
        for simulation in it:
            sequence = perc.random_sequence()
            simulation[...] = percolation_density(sequence)

    return densities


if __name__ == "__main__":

    size = 2048
    number_of_runs = 100
    rng = np.random.default_rng(seed=31415926)
    print(size)
    percolation_densities = run_simulation(size, number_of_runs, rng)

    np.savetxt(f"data_both/run{number_of_runs}_size{size}.txt", percolation_densities)
