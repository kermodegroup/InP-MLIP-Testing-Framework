
from active_model import *
from Utils.file_io import ase_read as read
import numpy as np
import matplotlib.pyplot as plt
from Utils.neb_core import do_NEB
import os
from ase.units import kB
from itertools import combinations
from ase.neighborlist import mic

marks = ["*"]


calc_names = plot_models[1:]
calcs = [get_model(name) for name in calc_names]


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

normalise = True


e_cut = 8.0 # eV

for i, defect in enumerate(defects):
    print(defect)
    for j, calc_name in enumerate(calc_names):
        calc = calcs[j]
        infile = "../Test_Results/" + calc_name + os.sep + "PDMigration/" + defect + "_final_ims.xyz"

        if os.path.exists(infile):
            ats = read(infile, index=":")

            Es = []

            for image in ats:
                image.calc = calc
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

            plt.plot(d, Es, label=calc_name, marker=".", color=f"C{j}")

    plt.title(titles[i])
    if normalise:
        plt.xlabel("Reaction Coordinate")
    else:
        plt.xlabel("Integrated Displacement (Ang)")

    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.savefig(f"..Test_Plots/PDMigration/{defect}.png")
    plt.clf()