from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

from estimators import jackknife

plt.style.use("ggplot")

data_files = Path("data_both/")


def get_size(file):
    file_name = str(file)
    size_index = file_name.index("size") + 4
    return int(file_name[size_index:-4])


data = {}

for file in data_files.iterdir():
    size = get_size(file)

    file_content = np.loadtxt(file)

    try:
        length = len(file_content)
    except TypeError:
        length = 1

    if data.get(size):
        if data[size]["runs"] < length:
            data[size]["runs"] = length
            data[size]["data"] = file_content

    else:
        data[size] = {"data": file_content, "runs": length}


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

        steps, prob = get_pi_curve(data[size]["data"])

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


# Analysis of exponent nu and critical point

new_data = {}


def erf_func(phi, delta, avg):
    return 0.5 + 0.5 * erf((phi - avg) / delta)


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
        return data.std(), data.mean()

    steps, prob = get_pi_curve(data)

    params, extras = curve_fit(erf_func, steps, prob)

    delta_err = jackknife(data, get_delta)
    avg_err = jackknife(data, get_avg)

    return params[0], params[1], delta_err, avg_err


for size in sorted(data):
    delta, avg, delta_err, avg_err = get_delta_and_avg(data[size]["data"], False)

    new_data[size] = {
        "delta": delta,
        "avg": avg,
        "delta_err": delta_err,
        "avg_err": avg_err,
    }

log_l = [np.log(size) for size in new_data]
log_delta = [np.log(1 / new_data[size]["delta"]) for size in new_data]
log_err = [
    -1 * new_data[size]["delta_err"] / new_data[size]["delta"] for size in new_data
]


linear_model = np.polyfit(log_l[:], log_delta[:], 1)
lin_func = np.poly1d(linear_model)


plt.errorbar(log_l, log_delta, yerr=log_err, linestyle="none", marker=".")
plt.plot(log_l, lin_func(log_l), label=rf"$1 / \nu= {linear_model[0].round(3)}$")
plt.title(r"Determinación del exponente crítico $\nu$")
plt.xlabel(r"$\log (L)$")
plt.ylabel(r"$\log ( \Delta(L)^{-1})$")
plt.legend()
plt.show()
# plt.savefig("images/nu_determination.png")

delta = [new_data[size]["delta"] for size in new_data]
avg = [new_data[size]["avg"] for size in new_data]
delta_err = [new_data[size]["delta_err"] for size in new_data]
avg_err = [new_data[size]["avg_err"] for size in new_data]

linear_model = np.polyfit(delta[4:], avg[4:], 1)
lin_func = np.poly1d(linear_model)


plt.close()
plt.errorbar(delta, avg, xerr=delta_err, yerr=avg_err, linestyle="none", marker=".")
plt.plot(delta, lin_func(delta), label=rf"$ p_{{crit}} = {linear_model[1].round(4)}$")
plt.title("Determinación del punto crítico")
plt.xlabel(r"$\Delta (L)$")
plt.ylabel(r"$p_{avg}(L)$")
plt.legend()
plt.show()
# plt.savefig("images/_crit_density_determination.png")

# graph_probabilities_vs_densities(data, True, "images/prob_density.png")
graph_probabilities_vs_densities(data)