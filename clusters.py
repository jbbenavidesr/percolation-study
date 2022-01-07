"""Cluster identification by using Hosen-Kopelman Algorithm"""
import numpy as np


def label_cluster(lattice: np.array) -> np.array:
    largest_label = 0
    clusters = np.zeros_like(lattice, np.int64)
    labels = np.arange(np.prod(lattice.shape))

    def find(x: int) -> int:
        y = x
        while labels[y] != y:
            y = labels[y]

        while labels[x] != x:
            z = labels[x]
            labels[x] = y
            x = z

        return y

    def union(x: int, y: int) -> None:
        labels[find(x)] = find(y)

    # TODO: Change the following to a more efficient way using numpy's methods
    for y, row in enumerate(lattice):
        for x, item in enumerate(row):
            if item:
                above = clusters[y - 1, x]
                left = clusters[y, x - 1]
                if y - 1 < 0:
                    above = 0
                if x - 1 < 0:
                    left = 0

                if left == 0 and above == 0:
                    largest_label += 1
                    clusters[y, x] = largest_label
                elif left != 0 and above == 0:
                    clusters[y, x] = find(left)
                elif left == 0 and above != 0:
                    clusters[y, x] = find(above)
                else:
                    union(left, above)
                    clusters[y, x] = find(left)

    with np.nditer(clusters, op_flags=["readwrite"]) as it:
        for cluster in it:
            if cluster != 0:
                cluster[...] = find(labels[cluster])

    return clusters


if __name__ == "__main__":
    from sitepercolation import run_square_site_simulation

    label_cluster(run_square_site_simulation(10000, 0.6))
