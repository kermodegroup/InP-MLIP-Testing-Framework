
from active_model import *
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import os
from ase.neighborlist import mic

marks = ["*"]


calc_names = plot_models[1:]


defects = [
    "30deg", "90deg"
]

titles = [
    "30 degree Partial Migration",
    "90 degree Partial Migration",
]

normalise = True


e_cut = 2.0 # eV

for i, defect in enumerate(defects):
    print(defect)
    for j, calc_name in enumerate(calc_names):
        infile = "../Test_Results/" + calc_name + os.sep + defect + "_Quadrupole_Migration_structs.xyz"

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

            plt.plot(d, Es, label=calc_name, marker=".", color=f"C{j}")

    plt.title(titles[i])
    if normalise:
        plt.xlabel("Reaction Coordinate")
    else:
        plt.xlabel("Integrated Displacement (Ang)")

    plt.ylabel("Energy (eV)")
    plt.legend()
    plt.savefig(f"../Test_Plots/{defect}_Migration.png")
    plt.clf()