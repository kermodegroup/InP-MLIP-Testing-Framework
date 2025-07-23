import matplotlib.pyplot as plt
from active_model import *
from Utils.plot_atoms import plot_atoms
from ase.io import read
import numpy as np
from matscipy.utils import get_structure_types
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import plot_config

plot_config.half_plot()

fig = plt.figure(figsize=(8, 11))

gs_30deg_top = GridSpec(1, 4, hspace=0.1, wspace=0.1)
gs_30deg_bottom = GridSpec(1, 3, hspace=0.1, wspace=0.1)

gs_90deg_top = GridSpec(1, 4, hspace=0.1, wspace=0.1)
gs_90deg_bottom = GridSpec(1, 3, hspace=0.1, wspace=0.1)

gs_legend = GridSpec(1, 1)

gs_30deg_top.update(left=0.02, right=0.98, top=0.94, bottom=0.74)
gs_30deg_bottom.update(left=0.14, right=0.86, top=0.82, bottom=0.62)

gs_90deg_top.update(left=0.02, right=0.98, top=0.53, bottom=0.33)
gs_90deg_bottom.update(left=0.14, right=0.86, top=0.41, bottom=0.21)

gs_legend.update(left=0.06, right=0.96, top=0.19, bottom=0.02)

ax_30deg = [plt.subplot(gs_30deg_top[j]) for j in range(4)] + [plt.subplot(gs_30deg_bottom[j]) for j in range(3)]
ax_90deg = [plt.subplot(gs_90deg_top[j]) for j in range(4)] + [plt.subplot(gs_90deg_bottom[j]) for j in range(3)]
legax = plt.subplot(gs_legend[0])


model_names = [model for model in plot_models if model != "DFT"]
nmodels = len(model_names)

dft_structs = [read("DFT_Reference/Quadrupoles/DFT_Quads_0/DFT_Quads_0.geom", index="-1"), read("DFT_Reference/Quadrupoles/DFT_Quads_1/DFT_Quads_1.geom", index="-1")]


disloc_names = ["30_degree_Partial", "90_degree_Partial"]
latex_names = ["$30^\circ$ Partial", "$90^\circ$ Partial"]

for j in range(2):
    # plt.clf()
    # fig, ax = plt.subplots(2, int(np.ceil((nmodels+1)/2)), figsize=(10, 8), gridspec_kw = {"hspace" : 0.1, "wspace" : 0.10})
    # plt.subplots_adjust(top=0.90, bottom=0.05, left=0.01, right=0.99)
    ax = [ax_30deg, ax_90deg][j]
    #ax = ax.flatten()


    # DFT
    atom_labels, struct_names, colors = get_structure_types(dft_structs[j], diamond_structure=True)
    atom_colors = [colors[atom_label] for atom_label in atom_labels]
    plot_atoms(ax[0], dft_structs[j], atom_colors, atom_labels)
    ax[0].text(0.5, 0.98, "RSCAN DFT", color="k", va="bottom", ha="center", transform=ax[0].transAxes)
    #ax[0].set_ylim(0, 46)
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    ax[0].axis("off")

    for i, model in enumerate(model_names):

        quads = read(f"../Test_Results/{model}/Quadrupole_structs.xyz", index=":")

        atom_labels, struct_names, colors = get_structure_types(quads[j], diamond_structure=True)
        atom_colors = [colors[atom_label] for atom_label in atom_labels]
        #plot_atoms(quads[j], ax=ax[i+1], colors=atom_colors)
        plot_atoms(ax[i+1], quads[j], atom_colors, atom_labels)

        if i < 3:
            y = 1.01
        else:
            y = -0.11
        ax[i+1].text(0.5, y, model, color="k", va="bottom", ha="center", transform=ax[i+1].transAxes)
        #ax[i+1].set_ylim(0, 46)
        #ax[i+1].set_xticks([])
        #ax[i+1].set_yticks([])
        ax[i+1].axis("off")

patches = []

labs = [struct_names[idx].replace(" (", "\n(") for idx in range(len(struct_names))]
labs[4] = "Hexagonal\nDiamond"

for idx in range(len(colors)):
    patches.append(mpatches.Patch(color=colors[idx], label=labs[idx]))

legax.axis("off")
legend = legax.legend(handles=patches, ncols=3, title="Atomic Coordination", fontsize=14, loc="center")

plt.setp(legend.get_title(),fontsize=18)

fig.text(0.5, 0.97, latex_names[0] + " Dislocation Quadrupole", color="k", ha="center", fontsize=20)
fig.text(0.5, 0.56, latex_names[1] + " Dislocation Quadrupole", color="k", ha="center", fontsize=20)
#plt.tight_layout()

plt.savefig(f"../Test_Plots/Quadrupole_Comparison.pdf")