import numpy as np


def jackknife(data, measure, n_groups=10):
    # divide list in n_group samples
    avg = len(data) / n_groups
    sample_list = []
    last = 0

    while last < len(data):
        sample_list.append(data[int(last) : int(last + avg)])
        last += avg

    # Get the list the measure in each group
    measure_list = []
    for i in range(n_groups):
        sample = []
        for j in range(n_groups):
            if j != i:
                sample += list(sample_list[j])

        measure_list.append(measure(sample))

    return np.mean(measure_list), 1.96 * np.std(measure_list)


def bootstrap(
    data: np.array,
    measure,
    confidence_factor: float = 1.96,
    sample_size: int = -1,
    number_of_samples: int = 10000,
    seed: int = 42,
):
    """Bootstrap resampling method for calculating error estimates"""
    rng = np.random.default_rng(seed=seed)

    if sample_size < 0:
        sample_size = len(data)

    measure_list = np.zeros(number_of_samples, dtype=float)

    with np.nditer(measure_list, op_flags=["readwrite"]) as it:
        for item in it:
            sample = rng.choice(data, size=sample_size)
            item[...] = measure(sample)

    return measure_list.mean(), confidence_factor * measure_list.std()
