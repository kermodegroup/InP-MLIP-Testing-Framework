import json
import os
import numpy as np
import matplotlib.pyplot as plt
from active_model import *
from matplotlib.gridspec import GridSpec
import plot_config

plot_config.half_plot()

model_names = [model for model in plot_models if model != "DFT"]



dir = "../Test_Results"

files = []
srcs = []

for name in model_names:
    fname = dir + os.sep + name + os.sep + "PointDefect_Formation_Energies.json"
    if os.path.exists(fname):
        files.append(fname)
        srcs.append(name)

files.append(dir + os.sep + "DFT/PointDefect_Formation_Energies.json")

defects_list = ['ZB_In_Monoanti', 'ZB_P_Monoanti', 'ZB_In_Monovac', 'ZB_P_Monovac', 'ZB_Interstitial_In_Octa', 'ZB_Interstitial_In_Tetra', 
                'ZB_Interstitial_P_Dumbbell']

nmodels = len(files) - 1

bar_sep = 1.0
bar_width = 0.8 / (nmodels)

formation_energies = {}
colour_lookup = {}

for i, file in enumerate(files):
    if i < len(srcs):
        src = srcs[i]
    else:
        src = "DFT" 
    colour_lookup[src] = "C" + str(i)
    with open(file) as f:
        data = json.load(f)

    for key in data.keys():
        if key in ["zb_E0", "wz_E0"]:
            continue

        if key in formation_energies.keys():
            formation_energies[key][src] = data[key]
        else:
            formation_energies[key] = {src: data[key]}

defects = sorted([key for key in formation_energies.keys()])

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 1, height_ratios=[5, 1])
ax = [fig.add_subplot(gs[i]) for i in range(2)]

#fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

dft_key = "DFT"

ax[0].set_xlim(-nmodels/2 * bar_width, (len(defects_list)-1) * bar_sep + nmodels/2 * bar_width)
dx = 0.015
for i, key in enumerate(defects_list):
    for j, model in enumerate(srcs):
        ax[0].bar(i * bar_sep + (j-nmodels/2) * bar_width, formation_energies[key][model]["E_form"],
                        color=colour_lookup[model], label=model, width=bar_width, align="edge", alpha=0.7)

    ax[0].bar(i * bar_sep - (nmodels/2 * bar_width) + dx, formation_energies[key]["DFT"]["E_form"], fill=False,
              color="k", label="RSCAN DFT", width=bar_width*nmodels - 2*dx, align="edge", linewidth=2)
d = {
    "ZB_In_Monoanti" : r"P$\rightarrow$In Antisite",
    "ZB_P_Monoanti" : r"In$\rightarrow$P Antisite",
    "ZB_Interstitial_In_Tetra" : "In Tetrahedral Interstitial",
    "ZB_Interstitial_In_Octa" : "In Octahedral Interstitial",
    "ZB_Interstitial_P_Dumbbell" : "P Dumbbell Interstitial",
    "ZB_In_Monovac" : r"In Vacancy",
    "ZB_P_Monovac" : r"P Vacancy",
}

defect_names = [d[defect] for defect in defects_list]


ax[0].set_xticks(np.arange(len(defects_list)) * bar_sep,
                    labels=defect_names, rotation=30, rotation_mode='anchor', ha="right")


ax[0].set_ylabel(r"Formation Energy (eV; $\mu_\text{In}=\mu_\text{P}$)")

handles, labels = ax[0].get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(
    zip(handles, labels)) if l not in labels[:i]]

zip = zip(*unique)

handles, labels = zip

ax[1].axis("off")
ax[1].legend(handles, labels, loc="center", ncols=4)
ax[0].set_box_aspect(1/2)

ax[0].axhline(0, color="k")

#ax[0].set_title("Formation Energies of Various Point Defects")
plt.tight_layout()
plt.savefig("../Test_Plots/PointDefects.pdf")
