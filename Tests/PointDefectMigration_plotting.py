
from active_model import *
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import os
from ase.units import kB
from itertools import combinations
from ase.neighborlist import mic

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




normalise = True

e_cuts = [2, 0.3, 2, 2, 1.2, 5, 2.5] # eV

os.makedirs("../Test_Plots/PDMigration", exist_ok=True)

for i, defect in enumerate(defects):
    e_cut = e_cuts[i]
    print(defect)
    for j, calc_name in enumerate(calc_names):
        infile = "../Test_Results/" + calc_name + os.sep + "PDMigration/" + defect + "_final_ims.xyz"

        if os.path.exists(infile):
            ats = read(infile, index=":")

            # if calc_name == "ACE":
            #     write(defect + "_Singlepoints.xyz", ats[::2])

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

            if print_barriers:
                print(defect, calc_name, np.max(Es))

            mask = np.abs(Es) < e_cut

            Es[~mask] = np.inf

            if normalise:

                d /= np.max(d)

            if calc_name == "ACE":
                d_dft = d[::2]

            plt.plot(d, Es, label=calc_name, marker=".", color=f"C{j}")

    ### DFT reference
    ### Relaxed Endpoints
    dft_start = defect_structs[defect_startnames[i]]
    dft_end = defect_structs[defect_endnames[i]]

    E_start = dft_start.get_potential_energy()

    if dft_end is not None:
        E_end = dft_end.get_potential_energy() - E_start
        plt.scatter([0, 1], [0, E_end], label="DFT Relaxation", color="k", marker="s", zorder=5, s=60)


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

    plt.scatter(d_dft, E_singles, label="DFT Singlepoint\nfrom ACE Structures", color="k", marker="x", zorder=5, s=60, linewidth=2)

    plt.title(titles[i])
    if normalise:
        plt.xlabel("Reaction Coordinate")
    else:
        plt.xlabel("Integrated Displacement (Ang)")

    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.savefig(f"../Test_Plots/PDMigration/{defect}.png")
    plt.clf()