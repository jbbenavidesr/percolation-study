"""Module for handling site-percolation"""
import numpy as np


class SitePercolation:
    """Object that stores data about site percolation lattice, for the moment just squares"""

    def __init__(self, length: int, rng: np.random.Generator = None, seed: int = 42):
        """
        Initialize the site percolation class with a random number generator and a lattice
        """

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed=seed)

        self.length = length

    def random_distribution(self, occupation_probability: float) -> np.array:
        """Get a random distribution with given occupation probability."""

        lattice = self.rng.random((self.length, self.length))

        return lattice < occupation_probability

    def random_sequence(self) -> np.array:
        """Ger a random sequence in which sites are occupied."""

        sequence = np.arange(self.length**2)

        self.rng.shuffle(sequence)

        return sequence

    @staticmethod
    def state_of_lattice(sequence: np.array, current_step: int) -> np.array:
        """Method for getting the distribution of the lattice at a given step of a sequence"""
        length_of_side = int(np.sqrt(len(sequence)))

        lattice = np.zeros_like(sequence, dtype=int)

        lattice[sequence[:current_step]] = 1

        return lattice.reshape((length_of_side, length_of_side))
