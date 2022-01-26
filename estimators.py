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
                for point in sample_list[j]:
                    sample.append(point)

        measure_list.append(measure(sample))

    return (n_groups - 1) / np.sqrt(n_groups) * np.std(measure_list)


def bootstrap(data, measure, sample_size=-1, n_samples=1000):
    n = len(data)
    if sample_size < 0:
        sample_size = int(len(data) * 0.9)

    measure_list = []

    for _ in range(n_samples):
        data_group = []
        for _ in range(sample_size):
            data_group.append(data[np.random.randint(0, n)])

        measure_list.append(measure(data_group))

    return np.mean(measure_list), np.std(measure_list)
