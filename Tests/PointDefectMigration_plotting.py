
from active_model import *
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import os
from ase.units import kB
from itertools import combinations
from ase.neighborlist import mic
import plot_config

plot_config.full_plot()

def plot_ax(ax, i):
    defect = defects[i]
    startname = defect_startnames[i]
    endname = defect_endnames[i]
    title = titles[i]
    e_cut = e_cuts[i]
    
    for j, calc_name in enumerate(calc_names):
        infile = "../Test_Results/" + calc_name + os.sep + "PDMigration/" + defect + "_final_ims.xyz"

        if os.path.exists(infile):
            ats = read(infile, index=":")

            Es = []

            for image in ats:
                Es.append(image.get_potential_energy())
            Es = np.array(Es)

            p0 = ats[0].positions

            d = np.zeros(len(ats))

            for idx in range(1, len(ats)):
                dR = mic(ats[idx].positions - ats[idx-1].positions, ats[idx].cell)
                d[idx] = d[idx-1] + np.sqrt((dR**2).sum())

            E_min = Es[0]

            Es -= E_min

            mask = np.abs(Es) < e_cut

            Es[~mask] = np.inf

            if normalise:
                d /= np.max(d)

            if calc_name == "ACE":
                d_dft = d[::2]

            ax.plot(d, Es, label=calc_name, marker=".", color=f"C{j}")

    ### DFT reference
    ### Relaxed Endpoints
    dft_start = defect_structs[startname]
    dft_end = defect_structs[endname]

    E_start = dft_start.get_potential_energy()

    if dft_end is not None:
        E_end = dft_end.get_potential_energy() - E_start
        ax.scatter([0, 1], [0, E_end], label="DFT Relaxation", color="k", marker="s", zorder=5)


    ### Singlepoints

    E_singles = [0] * 8

    for j in range(8):
        try:
            singlepoint = read(f"DFT_Reference/PD_Migration/Singlepoints/{defect}_Singlepoints_{j}/{defect}_Singlepoints_{j}.castep", index="0")
            E_singles[j] = singlepoint.get_potential_energy()
        except BaseException as e:
            print(type(e))
            print(defect, j, " issue")
            E_singles[j] = np.inf

    E_singles = np.array(E_singles)
    E_singles -= E_start

    ax.scatter(d_dft, E_singles, label="DFT Singlepoint\nfrom ACE Structures", color="k", marker="x", zorder=5, linewidth=2)

    ax.set_title(title)

    if normalise:
        ax.set_xlabel("Reaction Coordinate")
    else:
        ax.set_xlabel("Integrated Displacement (Ang)")

    ax.set_ylabel("Energy (eV)")

    return ax

marks = ["*"]

# Whether to plot the barrier heights for each calc + defect
print_barriers = True
print_barriers = False

calc_names = plot_models[1:]


defects = [
    "In_Interstitial_Migration_Tetra_Octa",
    "P_Interstitial_Migration_Dumbbell_Dumbbell+",
    "P_Interstitial_Migration_Dumbbell#_Dumbbell+",
    "In_Vacancy_Migration_15->60",
    "In_Vacancy_Migration_3->60",
    "P_Vacancy_Migration_5->3",
    "P_Vacancy_Migration_0->3"
]

titles = [
    "In Interstitial Migration\nTetrahedral -> Octahedral",
    "P Interstitial Migration\nDumbbell Rotation",
    "P Interstitial Migration\nDumbbell Migration",
    "In Vacancy Migration",
    "In Vacancy Migration\nP entering In site",
    "P Vacancy Migration\nIn entering P site",
    "P Vacancy Migration"
]

defect_startnames = [
    "ZB_Interstitial_In_Tetra",
    "ZB_Interstitial_P_Dumbbell",
    "ZB_Interstitial_P_Dumbbell+",
    "ZB_In_Monovac",
    "ZB_In_Monovac",
    "ZB_P_Monovac",
    "ZB_P_Monovac"
]

defect_endnames = [
    "ZB_Interstitial_In_Octa",
    "ZB_Interstitial_P_Dumbbell",
    "ZB_Interstitial_P_Dumbbell#",
    "ZB_In_Monovac",
    "P_Anti_In_Vac",
    "In_Anti_P_Vac",
    "ZB_P_Monovac",
]


defects_list = ['ZB_In_Monovac', 'ZB_P_Monovac', 'ZB_Interstitial_In_Octa', 'ZB_Interstitial_In_Tetra', 
                'ZB_Interstitial_P_Dumbbell']

defect_structs = {name : read(f"DFT_Reference/PointDefects/{name}.xyz", index="-1") for name in defects_list}
defect_structs[None] = None

defect_structs["P_Anti_In_Vac"] = read(f"DFT_Reference/PD_Migration/Endpoints/In_Vacancy_Antisite/In_Vacancy_Antisite.geom", index="-1")

defect_structs["In_Anti_P_Vac"] = read(f"DFT_Reference/PD_Migration/Endpoints/P_Vacancy_Antisite/P_Vacancy_Antisite.geom", index="-1")

defect_structs["ZB_Interstitial_P_Dumbbell+"] = read(f"DFT_Reference/PD_Migration/Endpoints/P_Interstitial_Dumbbell_plus/P_Interstitial_Dumbbell_plus.geom", index="-1")

defect_structs["ZB_Interstitial_P_Dumbbell#"] = read(f"DFT_Reference/PD_Migration/Endpoints/P_Interstitial_Dumbbell_hash/P_Interstitial_Dumbbell_hash.geom", index="-1")


In_Interstitial_idxs = [0]
In_Vac_idxs = [3, 4]
P_Interstitial_idxs = [1, 2]
P_Vac_idxs = [5, 6]



normalise = True

e_cuts = [2, 0.3, 2, 2, 1.2, 5, 2.5] # eV

os.makedirs("../Test_Plots/PDMigration", exist_ok=True)

######################
### Indium Defects ###
######################

fig, ax = plt.subplots(2, 2, figsize=(16, 12))

ax = ax.flatten()

# In Interstitial
plot_ax(ax[0], 0)

# In Vacancy
plot_ax(ax[2], 3)
plot_ax(ax[3], 4)

ax[1].axis("off")

handles, labels = ax[0].get_legend_handles_labels()
ax[1].legend(handles, labels, loc="upper left")

plt.tight_layout()
plt.savefig(f"../Test_Plots/In_Defect_Migration.pdf")

##########################
### Phosphorus Defects ###
##########################
plt.clf()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(16, 14))
plotgrid = GridSpec(2, 2, hspace=0.5, wspace=0.3)
leggrid = GridSpec(1, 1)

plotgrid.update(bottom=0.18)
leggrid.update(bottom=0.00, top=0.11)

ax = [plt.subplot(plotgrid[i, j]) for i in range(2) for j in range(2)]

legax = plt.subplot(leggrid[0])

# P Interstitials
plot_ax(ax[0], 2)
plot_ax(ax[1], 1)

# P Vacancies
plot_ax(ax[2], 6)
plot_ax(ax[3], 5)

handles, labels = ax[0].get_legend_handles_labels()

legax.axis("off")
legax.legend(handles, labels, ncols=4, loc="center left")

plt.savefig(f"../Test_Plots/P_Defect_Migration.pdf")