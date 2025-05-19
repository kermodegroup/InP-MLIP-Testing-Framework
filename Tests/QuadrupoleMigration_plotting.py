
from active_model import *
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
import os
from ase.neighborlist import mic

marks = ["*"]


#calc_names = plot_models[1:]
calc_names = ["_", "_", "MP0", "MPA", "ACE", "MACE"]

defects = [
    "30deg", "90deg"
]

titles = [
    "30 degree Partial Migration",
    "90 degree Partial Migration",
]

normalise = True

dft_startpoints = [read(f"DFT_Reference/Quadrupoles/DFT_Quads_{i}/DFT_Quads_{i}.geom", index="-1") for i in range(2)]
dft_endpoints = [read(f"DFT_Reference/QuadMigration/Endpoints/QuadRelaxEndpoints_{i}/QuadRelaxEndpoints_{i}.geom", index="-1") for i in range(2)]


dft_endpoints[-1] = read(f"DFT_Reference/QuadMigration/Endpoints/90deg_Quadrupole_Migration_Endpoint/90deg_Quadrupole_Migration_Endpoint.geom", index="-1")


e_cut = 2.0 # eV

for i, defect in enumerate(defects):
    print(defect)
    for j, calc_name in enumerate(calc_names):
        infile = "../Test_Results/" + calc_name + os.sep + defect + "_Quadrupole_Migration_structs.xyz"

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

            mask = np.abs(Es) < e_cut

            Es[~mask] = np.inf

            if normalise:

                d /= np.max(d)

            if calc_name == "ACE":
                d_dft = d[::2]

            plt.plot(d, Es, label=calc_name, marker=".", color=f"C{j}")


    ### Endpoints

    dft_start = dft_startpoints[i]
    dft_end = dft_endpoints[i]

    E_start = dft_start.get_potential_energy()
    E_end = dft_end.get_potential_energy()

    plt.scatter([0, 1], [0, E_end - E_start], marker="s", color="k", label="DFT Relaxation")

    ### Singlepoints

    E_singles = [0] * 8

    for j in range(8):
        try:
            singlepoint = read(f"DFT_Reference/QuadMigration/Singlepoints/{defect}_Singlepoints_{j}/{defect}_Singlepoints_{j}.castep", index="0")
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
    plt.savefig(f"../Test_Plots/{defect}_Migration.png")
    plt.clf()