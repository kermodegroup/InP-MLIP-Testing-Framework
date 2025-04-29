from active_model import *
import matplotlib.colors as mcolors
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from get_true_values import *

data_src = savedir = data_dir + os.sep + active_model_name + os.sep + "TT_RMSE"

with open(data_src + os.sep + "True_Vals.json") as f:
    true_vals = json.load(f)


colours = list(mcolors.TABLEAU_COLORS.values())
markers = ["x", "o", "s", ".", "*", "+", "D", "^", "<", ">", "v", "H", "d"]

plot_colours = colours

scatter_colours_markers = []

for marker in markers:
    for colour in colours:
        scatter_colours_markers.append((colour, marker))

def gen_plots(raw_data, calc_name):
    config_types = [t for t in raw_data["energy"].keys()
                    if "Isolated" not in t]

    fig, ax = plt.subplots(3, 3, figsize=(16, 8))
    ax = ax

    e_min = np.inf
    e_max = 0
    f_min = np.inf
    f_max = 0
    v_min = np.inf
    v_max = 0

    ne = 0
    nf = 0
    nv = 0

    for config_type in config_types:
        ne += len(raw_data["energy"][config_type])
        nf += len(raw_data["forces"][config_type])
        nv += len(raw_data["stresses"][config_type])

        e_min = min([e_min, min(raw_data["energy"][config_type])])
        e_max = max([e_max, max(raw_data["energy"][config_type])])
        f_min = min([f_min, min(raw_data["forces"][config_type])])
        f_max = max([f_max, max(raw_data["forces"][config_type])])

        if len(raw_data["stresses"][config_type]):
            v_min = min([v_min, min(raw_data["stresses"][config_type])])
            v_max = max([v_max, max(raw_data["stresses"][config_type])])

    e_bins = np.linspace(e_min, e_max, int(ne / 100))
    f_bins = np.linspace(f_min, f_max, int(nf / 1000))
    v_bins = np.linspace(v_min, v_max, int(nv / 100))

    bins = {
        "energy": e_bins,
        "forces": f_bins,
        "stresses": v_bins
    }

    rmses = np.zeros(3)
    for j, obs in enumerate(["energy", "forces", "stresses"]):
        accumulation = []
        for i, config_type in enumerate(config_types):
            truth = true_vals[obs][config_type]
            err = raw_data[obs][config_type]

            accumulation.extend(err)

            colour, marker = scatter_colours_markers[i]
            ax[j, 0].scatter(truth, err, color=colour,
                                marker=marker, alpha=0.2, label=config_type)
            
        ax[j, 1].hist(accumulation, bins=bins[obs], orientation="horizontal")
        rmses[j] = np.sqrt(np.average(np.array(accumulation)**2))

    ax[0, 1].text(0.99, 0.99, f'Energy Observations: {format(ne, ",d")} \nEnergy RMSE={"%s" % float("%.3g" % rmses[0])} meV/Atom',
                  horizontalalignment='right', verticalalignment='top', transform=ax[0, 1].transAxes)
    ax[1, 1].text(0.99, 0.99, f'Force Observations: {format(nf, ",d")} \nForce RMSE={"%s" % float("%.3g" % rmses[1])} meV/A',
                  horizontalalignment='right', verticalalignment='top', transform=ax[1, 1].transAxes)
    ax[2, 1].text(0.99, 0.99, f'Stress Observations: {format(nv, ",d")} \nStress RMSE={"%s" % float("%.3g" % rmses[2])} MPa',
                  horizontalalignment='right', verticalalignment='top', transform=ax[2, 1].transAxes)

    ax[0, 0].set_xlabel("True Binding Energy (eV/Atom)")
    ax[1, 0].set_xlabel("True Force (eV/A)")
    ax[2, 0].set_xlabel("True Stress (GPa)")

    ax[0, 0].set_ylabel("Energy Error (meV/Atom)")
    ax[1, 0].set_ylabel("Force Error (meV/A)")
    ax[2, 0].set_ylabel("Stress Error (MPa)")

    handles, labels = ax[0, 0].get_legend_handles_labels()

    ax = ax.T
    ax[2, 0].axis("off")
    ax[2, 1].axis("off")
    ax[2, 2].axis("off")

    leg = ax[2, 0].legend(handles, labels)

    for lh in leg.legend_handles:
        lh.set_alpha(1)

    return fig, ax


def gen_plots_individual(raw_data, calc_name):
    config_types = [t for t in raw_data["energy"].keys()
                    if "Isolated" not in t]

    e_min = np.inf
    e_max = 0
    f_min = np.inf
    f_max = 0
    v_min = np.inf
    v_max = 0

    ne = 0
    nf = 0
    nv = 0

    for config_type in config_types:
        ne += len(raw_data["energy"][config_type])
        nf += len(raw_data["forces"][config_type])
        nv += len(raw_data["stresses"][config_type])

        e_min = min([e_min, min(raw_data["energy"][config_type])])
        e_max = max([e_max, max(raw_data["energy"][config_type])])
        f_min = min([f_min, min(raw_data["forces"][config_type])])
        f_max = max([f_max, max(raw_data["forces"][config_type])])

        if len(raw_data["stresses"][config_type]):
            v_min = min([v_min, min(raw_data["stresses"][config_type])])
            v_max = max([v_max, max(raw_data["stresses"][config_type])])

    e_bins = np.linspace(e_min, e_max, int(ne / 100))
    f_bins = np.linspace(f_min, f_max, int(nf / 1000))
    v_bins = np.linspace(v_min, v_max, int(nv / 1000))

    bins = {
        "energy": e_bins,
        "forces": f_bins,
        "stresses": v_bins
    }

    rmses = np.zeros(3)
    ns = np.zeros(3, dtype=int)

    divisors = [10, 100, 100]
    for i, config_type in enumerate(config_types):
        plt.clf()

        fig, ax = plt.subplots(3, 2, figsize=(16, 8))
        ax = ax

        for j, obs in enumerate(["energy", "forces", "stresses"]):

            truth = true_vals[obs][config_type]
            err = raw_data[obs][config_type]

            colour, marker = scatter_colours_markers[i]

            ax[j, 0].scatter(truth, err)

            ax[j, 1].hist(err, bins=max(
                [1, int(len(err)/divisors[j])]), orientation="horizontal")
            rmses[j] = np.sqrt(np.average(np.array(err)**2))

            ns[j] = len(raw_data[obs][config_type])

        ax[0, 1].text(0.99, 0.99, f'Energy Observations: {format(ns[0], ",d")} \nEnergy RMSE={"%s" % float("%.3g" % rmses[0])} meV/Atom',
                      horizontalalignment='right', verticalalignment='top', transform=ax[0, 1].transAxes)
        ax[1, 1].text(0.99, 0.99, f'Force Observations: {format(ns[1], ",d")} \nForce RMSE={"%s" % float("%.3g" % rmses[1])} meV/A',
                      horizontalalignment='right', verticalalignment='top', transform=ax[1, 1].transAxes)
        ax[2, 1].text(0.99, 0.99, f'Stress Observations: {format(ns[2], ",d")} \nStress RMSE={"%s" % float("%.3g" % rmses[2])} MPa',
                      horizontalalignment='right', verticalalignment='top', transform=ax[2, 1].transAxes)

        ax[0, 0].set_xlabel("True Binding Energy (eV/Atom)")
        ax[1, 0].set_xlabel("True Force (eV/A)")
        ax[2, 0].set_xlabel("True Stress (GPa)")

        ax[0, 0].set_ylabel("Energy Error (meV/Atom)")
        ax[1, 0].set_ylabel("Force Error (meV/A)")
        ax[2, 0].set_ylabel("Stress Error (MPa)")

        plt.tight_layout()
        plt.title(config_type)

        plt.savefig("../Test_Plots/" + os.sep + calc_name + os.sep + "TT_Config_Types" +
                    os.sep + "TT_{}_Plot".format(config_type))


calc = active_model_name

os.makedirs("../Test_Plots/" + os.sep + calc +
                os.sep + "TT_Config_Types", exist_ok=True)

with open(data_src + os.sep + "RAW_TT_DIFFS.json") as f:
    raw_data = json.load(f)

fig, ax = gen_plots(raw_data, calc)

plt.tight_layout()
plt.savefig("../Test_Plots/" + calc + os.sep + "TT_Errs.png")

gen_plots_individual(raw_data, calc)
