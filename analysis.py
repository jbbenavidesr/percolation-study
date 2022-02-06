from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

from estimators import jackknife, bootstrap

# plt.style.use("seaborn")


def get_size(file):
    file_name = str(file)
    size_index = file_name.index("size") + 4
    return int(file_name[size_index:-4])


def load_data(path: str or Path) -> dict:
    data_files = Path(path)

    data = {}

    for file in data_files.iterdir():
        size = get_size(file)

        file_content = np.loadtxt(file)

        length = len(file_content)

        if data.get(size):
            if len(data[size]) < length:
                data[size] = file_content

        else:
            data[size] = file_content

    return data


def get_pi_curve(data):
    steps = np.linspace(0, 1, 150)

    prob = np.zeros_like(steps)

    for i, step in enumerate(steps):
        count = (data < step).sum()
        prob[i] = count / len(data)

    return steps, prob


def graph_probabilities_vs_densities(data, save=False, filename="graph.png"):
    plt.figure(figsize=(15, 8))

    for size in sorted(data):

        steps, prob = get_pi_curve(data[size])

        plt.plot(steps, prob, label=rf"$L = {size}$")

    plt.xlabel(r"$p$")
    plt.ylabel(r"$\Pi$")
    plt.title(
        "Probabilidad de encontrar un cluster percolante en función de la densidad para diferentes tamaños del sistema."
    )
    plt.legend()

    if save:
        plt.savefig(filename)
    else:
        plt.show()


def erf_func(phi, delta, avg):
    return 0.5 + 0.5 * erf((phi - avg) / delta)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def get_delta(data):
    steps, prob = get_pi_curve(data)

    params, extras = curve_fit(erf_func, steps, prob)

    return params[0]


def get_avg(data):
    steps, prob = get_pi_curve(data)

    params, extras = curve_fit(erf_func, steps, prob)

    return params[1]


def get_delta_and_avg(data, simple=True) -> tuple:
    if simple:
        delta, delta_err = bootstrap(data, np.std)
        avg, avg_err = bootstrap(data, np.mean)

    else:

        delta, delta_err = bootstrap(data, get_delta)
        avg, avg_err = bootstrap(data, get_avg)

    return delta, avg, delta_err, avg_err


def linear_fit_error(
    x, y, x_err=None, y_err=None, number_of_samples=1000, error_factor=1.96
):
    if not x_err and not y_err:
        return np.polyfit(x, y, 1)

    rng = np.random.default_rng(8993)
    slopes = np.zeros(number_of_samples, dtype=float)
    intercepts = np.zeros(number_of_samples, dtype=float)

    if not x_err:
        for i in range(number_of_samples):
            y_samples = rng.normal(y, y_err)
            lin_model = np.polyfit(x, y_samples, 1)
            slopes[i] = lin_model[0]
            intercepts[i] = lin_model[1]

    elif not y_err:
        for i in range(number_of_samples):
            x_samples = rng.normal(x, x_err)
            lin_model = np.polyfit(x_samples, y, 1)
            slopes[i] = lin_model[0]
            intercepts[i] = lin_model[1]

    else:
        for i in range(number_of_samples):
            x_samples = rng.normal(x, x_err)
            y_samples = rng.normal(y, y_err)
            lin_model = np.polyfit(x_samples, y_samples, 1)
            slopes[i] = lin_model[0]
            intercepts[i] = lin_model[1]

    return (
        [slopes.mean(), intercepts.mean()],
        1.96 * slopes.std(),
        1.96 * intercepts.std(),
    )


if __name__ == "__main__":

    path = "data/jigsaw_borders/"

    data = load_data(path)

    new_data = {}

    for size in sorted(data):
        delta, avg, delta_err, avg_err = get_delta_and_avg(data[size], simple=True)

        new_data[size] = {
            "delta": delta,
            "avg": avg,
            "delta_err": delta_err,
            "avg_err": avg_err,
        }

    log_l = [np.log(size) for size in new_data]
    log_delta = [np.log(1 / new_data[size]["delta"]) for size in new_data]
    log_err = [
        np.abs(-1 * new_data[size]["delta_err"] / new_data[size]["delta"])
        for size in new_data
    ]

    linear_model, slope_err, intercept_err = linear_fit_error(
        log_l[2:], log_delta[2:], y_err=log_err[2:]
    )
    lin_func = np.poly1d(linear_model)

    print(slope_err, intercept_err)

    plt.errorbar(log_l, log_delta, yerr=log_err, linestyle="none", marker=".")
    plt.plot(
        log_l,
        lin_func(log_l),
        label=rf"$1 / \nu= {linear_model[0].round(3)} \pm {slope_err.round(3)}$",
    )
    plt.title(r"Determinación del exponente crítico $\nu$")
    plt.xlabel(r"$\log (L)$")
    plt.ylabel(r"$\log ( \Delta(L)^{-1})$")
    plt.legend()
    plt.show()
    # plt.savefig("images/nu_jigsaw.png")

    delta = [new_data[size]["delta"] for size in new_data]
    avg = [new_data[size]["avg"] for size in new_data]
    delta_err = [new_data[size]["delta_err"] for size in new_data]
    avg_err = [new_data[size]["avg_err"] for size in new_data]

    linear_model, slope_err, intercept_err = linear_fit_error(
        delta[:], avg[:], x_err=delta_err[:], y_err=avg_err[:]
    )
    lin_func = np.poly1d(linear_model)

    print(slope_err, intercept_err)

    plt.close()
    plt.errorbar(delta, avg, xerr=delta_err, yerr=avg_err, linestyle="none", marker=".")
    plt.plot(
        delta,
        lin_func(delta),
        label=rf"$ p_{{crit}} = {linear_model[1].round(4)} \pm {intercept_err.round(4)}$",
    )
    plt.title("Determinación del punto crítico")
    plt.xlabel(r"$\Delta (L)$")
    plt.ylabel(r"$p_{avg}(L)$")
    plt.legend()
    plt.show()
    # plt.savefig("images/crit_density_jigsaw.png")

    plt.close()
    # graph_probabilities_vs_densities(data, True, "images/prob_density_jigsaw.png")
    graph_probabilities_vs_densities(data)

    log_l = [np.log(size) for size in new_data]
    log_avg = [np.log(1 / new_data[size]["avg"]) for size in new_data]
    log_err = [
        np.abs(-1 * new_data[size]["avg_err"] / new_data[size]["avg"])
        for size in new_data
    ]

    linear_model, slope_err, intercept_err = linear_fit_error(
        log_l[:], log_avg[:], y_err=log_err[:]
    )
    lin_func = np.poly1d(linear_model)
    print(slope_err, intercept_err)

    plt.close()
    plt.errorbar(log_l, log_avg, yerr=log_err, linestyle="none", marker=".")
    plt.plot(
        log_l,
        lin_func(log_l),
        label=rf"$\beta /\nu= {linear_model[0].round(3)} \pm {slope_err.round(3)}$",
    )
    plt.title(r"Determinación del exponente crítico $\beta / \nu$")
    plt.xlabel(r"$\log (L)$")
    plt.ylabel(r"$\log ( p_{avg}^{-1})$")
    plt.legend()
    plt.show()
    # plt.savefig("images/betta_jigsaw.png")

    plt.close()

    steps = np.linspace(0, 1, 150)

    for size in sorted(data):
        fig, ax1 = plt.subplots()
        ax1.hist(
            data[size],
            bins=int(np.sqrt(len(data[size])) + 2),
            density=True,
            histtype="step",
        )

        ax2 = ax1.twinx()

        ax2.plot(steps, gaussian(steps, new_data[size]["avg"], new_data[size]["delta"]))

        plt.title(size)
        plt.show()
