from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

encuts = [600, 750, 900, 1050, 1200]
N_k = np.array([4, 6, 8])


ref_cut = 1500
ref_k = 12

ref_ats = read(f"DFT_Files/{ref_cut}_{ref_k}/{ref_cut}_{ref_k}.castep")

E_ref = ref_ats.get_potential_energy()
F_ref = ref_ats.get_forces()

Es = np.zeros((len(encuts), len(N_k)))
Fs = np.zeros_like(Es)

for i, encut in enumerate(encuts):
    for j, N in enumerate(N_k):
        ats = read(f"DFT_Files/{encut}_{N}/{encut}_{N}.castep")

        Es[i, j] = (ats.get_potential_energy() - E_ref) / len(ats)

        Fs[i, j] = np.max(np.linalg.norm(ats.get_forces() - F_ref, axis=-1))

im = plt.imshow(Es.T * 1000)

plt.colorbar(im, label="Energy Error (meV/Atom)")

plt.xticks(np.arange(len(encuts)), encuts)
plt.xlabel("Cutoff Energy (eV)")

plt.yticks(np.arange(len(N_k)), N_k)
plt.ylabel("K-point Grid Size")

plt.title(f"Convergence of Total Energy Error with DFT settings\nE_cut = 900eV N_ k = 6 -> E_err = {1000*Es[2, 1]:.3f} meV/Atom")

plt.tight_layout()

plt.savefig("E_Converge.png")

plt.clf()


im = plt.imshow(Fs.T * 1000)

plt.colorbar(im, label="Max Force Error (meV/Ang)")


plt.xticks(np.arange(len(encuts)), encuts)
plt.xlabel("Cutoff Energy (eV)")

plt.yticks(np.arange(len(N_k)), N_k)
plt.ylabel("K-point Grid Size")

plt.title(f"Convergence of Max Force Error with DFT settings\nE_cut = 900eV N_ k = 6 -> F_err = {1000*Fs[2, 1]:.3f} meV/Ang")

plt.tight_layout()

plt.savefig("F_Converge.png")