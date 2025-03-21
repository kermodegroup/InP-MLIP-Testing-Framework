import matplotlib.pyplot as plt
from active_model import *
from plot_atoms import plot_atoms
from ase.io import read
import numpy as np
from matscipy.utils import get_structure_types
import matplotlib.patches as mpatches

model_names = [model for model in plot_models if model != "DFT"]
nmodels = len(model_names)

dft_structs = [read("DFT_Quadrupoles/DFT_Quads_0/DFT_Quads_0.geom", index="-1"), read("DFT_Quadrupoles/DFT_Quads_1/DFT_Quads_1.geom", index="-1")]


disloc_names = ["30_degree_Partial", "90_degree_Partial"]
latex_names = ["$30^\circ$ Partial", "$90^\circ$ Partial"]

for j in range(2):
    plt.clf()
    fig, ax = plt.subplots(2, int(np.ceil((nmodels+1)/2)), figsize=(10, 8), gridspec_kw = {"hspace" : 0.1, "wspace" : 0.10})
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.01, right=0.99)
    ax = ax.flatten()


    # DFT
    atom_labels, struct_names, colors = get_structure_types(dft_structs[j], diamond_structure=True)
    atom_colors = [colors[atom_label] for atom_label in atom_labels]
    plot_atoms(ax[0], dft_structs[j], atom_colors, atom_labels)
    ax[0].text(0.5, 1.0, "RSCAN DFT", color="k", va="bottom", ha="center", transform=ax[0].transAxes, fontsize=15)
    #ax[0].set_ylim(0, 46)
    #ax[0].set_xticks([])
    #ax[0].set_yticks([])
    ax[0].axis("off")

    for i, model in enumerate(model_names):

        quads = read(f"../Saved_Data/{model}/Quadrupole_structs.xyz", index=":")

        atom_labels, struct_names, colors = get_structure_types(quads[j], diamond_structure=True)
        atom_colors = [colors[atom_label] for atom_label in atom_labels]
        #plot_atoms(quads[j], ax=ax[i+1], colors=atom_colors)
        plot_atoms(ax[i+1], quads[j], atom_colors, atom_labels)
        ax[i+1].text(0.5, 1.0, model, color="k", va="bottom", ha="center", transform=ax[i+1].transAxes, fontsize=15)
        #ax[i+1].set_ylim(0, 46)
        #ax[i+1].set_xticks([])
        #ax[i+1].set_yticks([])
        ax[i+1].axis("off")

    if 2 * int(np.ceil((nmodels+1)/2)) != nmodels + 1:
        ax[-1].axis("off")
        patches = []
        for idx in range(len(colors)):
            patches.append(mpatches.Patch(color=colors[idx], label=struct_names[idx].replace("(", "\n(")))
        ax[-1].legend(handles=patches, loc=(0, 0.2))

    fig.suptitle(latex_names[j] + " Dislocation Quadrupole", color="k", size=20)
    #plt.tight_layout()

    plt.savefig(f"../Plots/{disloc_names[j]}_Comparison.png", dpi=300)