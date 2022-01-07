import numpy as np


def run_square_site_simulation(
    L: int, p: float, rng: np.random.Generator = None, seed: int = 42
) -> np.array:
    """Site percolation square lattice with side length L and probability p"""

    if not rng:
        rng = np.random.default_rng(seed=seed)

    lattice = rng.random((L, L))

    return lattice <= p


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    print(run_square_site_simulation(5, 0.5, rng) * 1)
