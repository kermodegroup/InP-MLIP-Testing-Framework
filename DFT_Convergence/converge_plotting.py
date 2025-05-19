from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

encuts = [600, 750, 900, 1050, 1200]
N_k = np.array([5, 6, 7, 8, 9])


ref_cut = 1500
ref_k = 12

ref_ats = read(f"{ref_cut}_{ref_k}/{ref_cut}_{ref_k}.castep")

E_ref = ref_ats.get_potential_energy()
F_ref = ref_ats.get_forces()

Es = np.zeros((len(encuts), len(N_k)))
Fs = np.zeros_like(Es)

for i, encut in enumerate(encuts):
    for j, N in enumerate(N_k):
        ats = read(f"{encut}_{N}/{encut}_{N}.castep")

        Es[i, j] = (ats.get_potential_energy() - E_ref) / len(ats)

        Fs[i, j] = np.max(np.linalg.norm(ats.get_forces() - F_ref, axis=-1))

im = plt.imshow(Es * 1000)

plt.colorbar(im, label="Energy Error (meV/Atom)")

plt.yticks(np.arange(len(encuts)), encuts)
plt.ylabel("Cutoff Energy (eV)")

plt.xticks(np.arange(len(N_k)), N_k)
plt.xlabel("K-point Grid Size")

plt.title(f"Convergence of Total energy with DFT settings\nE_cut = 900eV N_ k = 7 -> E_err = {1000*Es[2, 2]:.3f} meV/Atom")

plt.savefig("E_Converge.png")

plt.clf()


im = plt.imshow(Fs * 1000)

plt.colorbar(im, label="Max Force Error (meV/Ang)")


plt.yticks(np.arange(len(encuts)), encuts)
plt.ylabel("Cutoff Energy (eV)")

plt.xticks(np.arange(len(N_k)), N_k)
plt.xlabel("K-point Grid Size")


plt.title(f"Convergence of Total energy with DFT settings\nE_cut = 900eV N_ k = 7 -> F_err = {1000*Fs[2, 2]:.3f} meV/Ang")

plt.savefig("F_Converge.png")