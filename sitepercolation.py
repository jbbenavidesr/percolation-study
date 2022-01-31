import numpy as np


def run_square_site_simulation(
    L: int, p: float, rng: np.random.Generator = None, seed: int = 42
) -> np.array:
    """Site percolation square lattice with side length L and probability p"""

    if not rng:
        rng = np.random.default_rng(seed=seed)

    lattice = rng.random((L, L))

    return lattice < p


def get_random_sequence(
    L: int, rng: np.random.Generator = None, seed: int = 42
) -> np.array:
    """Get a random sequence that resembles the sequence that will come from the puzzles"""

    if not rng:
        rng = np.random.default_rng(seed=seed)

    sequence = np.arange(L * L)

    rng.shuffle(sequence)

    return sequence



def get_state_from_sequence(history: np.array, step: int, borders: bool = True) -> np.array:
    l_squared = len(history)
    l = int(np.sqrt(l_squared))

    lattice = np.zeros(l_squared, dtype=int)

    lattice[history[:step]] = 1

    lattice = lattice.reshape((l, l))

    if borders:
        return lattice
    
    return lattice[1:-1, 1:-1]
    
    


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    # print(run_square_site_simulation(5, 0.5, rng) * 1)
    seq = get_random_sequence(4, rng)

    print(seq)
    print(get_state_from_sequence(seq, 2))
    print(get_state_from_sequence(seq, 6))
