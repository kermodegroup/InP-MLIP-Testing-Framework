import matplotlib.pyplot as plt
from scipy.constants import h, e
import numpy as np
from ase.io import read as ase_read
from ase.phonons import Phonons
from active_model import *W

calc_names = plot_models[1:]

eV_to_THz = h * 1E12 / e

emax = 12 # THz
delta = 0.1
mul = 8
bulk = ase_read("Phonon_bulk.xyz")

fig, ax = plt.subplot_mosaic("AABC", figsize=(12, 6))
ax, dosax, legax = ax.values()

ax.set_ylim(0, emax)
ax.set_ylabel("Frequency (THz)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)


dosax.sharey(ax)
# dosax.set_yticks([])
# dosax.set_xticks([])
dosax.set_xlabel("Density of States", fontsize=14)
dosax.tick_params(labelleft=False, labelbottom=False)


legax.axis("off")

path = bulk.cell.bandpath('GXGL', npoints=100)

for c, calc_name in enumerate(calc_names):
    calc = None
    #calc = get_model(calc_name)
    #bulk.calc = calc
    name = f"../Test_Results/{calc_name}/Phonons"
    print(calc_name, name)
    ph = Phonons(bulk, calc, supercell=[mul]*3, delta=delta, name=name)
    ph.read(acoustic=True)
    #ph.clean()


    bs = ph.get_band_structure(path)
    coords, point_coords, point_names = path.get_linear_kpoint_axis()
    dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=300, width=1e-3)

    energies = np.array(bs._energies)
    dos_weights = np.array(dos.get_weights())

    kpath = bs._path.kpts
    dos_energies = dos.get_energies()



    E_means = energies[0, :, :]

    nbands = E_means.shape[1]


    for i in range(nbands):
        ax.plot(coords, E_means[:, i] / eV_to_THz, color=f"C{c}", label=calc_name)

    dosax.plot(dos_weights/np.sum(dos_weights), dos_energies / eV_to_THz, color=f"C{c}")

xticklabels = [
    "$\Gamma$", "X", "$\Gamma$", "L"
]

#for i, point in enumerate(xticklabels):
#    ax.text(point_coords[i], -0.1, point, fontsize=14)

print(point_coords)

ax.set_xticks(point_coords, xticklabels, fontsize=14)

# DATA FROM Borcherds, Alfrey, Saunderson & Woods

wavevectors = np.array([
    [0.0, 0, 0],  # Gamma
    [0.1, 0, 0],
    [0.2, 0, 0],
    [0.3, 0, 0],
    [0.4, 0, 0],
    [0.5, 0, 0],
    [0.6, 0, 0],
    [0.7, 0, 0],
    [0.8, 0, 0],
    [0.9, 0, 0],
    [1.0, 0, 0],  # X
    [1.0, 1.0, 0.0],
    [0.9, 0.9, 0],
    [0.8, 0.8, 0],
    [0.7, 0.7, 0],
    [0.6, 0.6, 0],
    [0.5, 0.5, 0],
    [0.4, 0.4, 0],
    [0.3, 0.3, 0],
    [0.2, 0.2, 0],
    [0.1, 0.1, 0],
    [0.0, 0, 0],  # Gamma
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2],
    [0.3, 0.3, 0.3],
    [0.4, 0.4, 0.4],
    [0.5, 0.5, 0.5]  # L
])

frequencies = np.array([
    [0, 0, 9.2, 10.38],  # Gamma
    [0.5, 1.02, -1, -1],
    [0.95, 1.65, 9.5, -1],
    [-1, 2.60, 9.4, 10.3],
    [1.80, -1, 9.45, -1],
    [-1, 3.75, 9.35, 10.5],
    [1.90, 4.45, -1, 10.5],
    [-1, 4.75, 9.55, 10.2],
    [-1, -1, 9.70, 9.90],
    [-1, -1, -1, 9.85],
    [2.05, 5.8, 9.70, 9.95],  # X
    [-1, -1, 9.7, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [2.30, 4.55, -1, -1],
    [-1, 4.28, -1, -1],
    [2.25, 3.82, -1, -1],
    [2.00, 3.19, -1, -1],
    [1.50, 2.34, -1, -1],
    [0.72, 1.28, 9.4, -1],
    [0, 0, 9.20, -1],  # Gamma
    [-1, -1, 9.3, 10.3],
    [0.69, 1.48, 9.5, 10.5],
    [1.30, 2.95, 9.4, -1],
    [1.53, 4.15, 9.3, 10.4],
    [1.72, 4.70, 9.6, 10.2],
    [1.65, 5.00, 9.5, 10.2]  # L
])


freq_errs = np.array([
    [0, 0, 0.2, 0],  # Gamma
    [0.01, 0.05, 0, 0],
    [0.05, 0.10, 0.30, 0],
    [0, 0.1, 0.2, 0.2],
    [0.05, 0, 0.15, 0],
    [0, 0.1, 0.1, 1.0],
    [0.1, 0.05, 0, 0.2],
    [0, 0.1, 0.2, 0.2],
    [0, 0, 0.2, 0.2],
    [0, 0, 0, 0.15],
    [0.1, 0.3, 0.2, 0.1],  # X
    [0, 0, 0.3, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0.05, 0.1, 0, 0],
    [0, 0.1, 0, 0],
    [0.1, 0.05, 0, 0],
    [0.2, 0.05, 0, 0],
    [0.1, 0.02, 0, 0],
    [0.02, 0.02, 0.3, 0],
    [0, 0, 0.2, 0],  # Gamma
    [0, 0, 0.3, 0.3],
    [0.02, 0.03, 0.2, 0.3],
    [0.05, 0.05, 0.1, 0],
    [0.03, 0.1, 0.2, 0.2],
    [0.05, 0.15, 0.2, 0.3],
    [0.02, 0.1, 0.15, 0.3]  # L
])


#frequencies *= h * 1E12 / e
#freq_errs *= h * 1E12 / e


frequencies = frequencies  # [:21, :]
freq_errs = freq_errs  # [:21, :]

# Plot Expt data
xlims = ax.get_xlim()

G1, X, G2, L = point_coords

G1_idx = 0
X_idx = 11
G2_idx = 21
L_idx = 27

x = np.zeros(frequencies.shape[0])

x[G1_idx:X_idx] = np.linspace(G1, X, X_idx - G1_idx)
x[X_idx:G2_idx] = np.linspace(X, G2, G2_idx - X_idx + 1)[1:]
x[G2_idx:L_idx] = np.linspace(G2, L, L_idx - G2_idx + 1)[1:]

#ax.scatter(x, frequencies, marker="x", color="k")
for i in range(4):
    ax.errorbar(x, frequencies[:, i], yerr=freq_errs[:, i],
                color="k", marker="x", linestyle="None", label="Experiment \n(Borcherds 1975)")
    # pass


ax.text((point_coords[1] + point_coords[0])/2, -0.1, r"($\zeta$, 0, 0)", fontsize=12, ha="center", va="top")
ax.text((point_coords[2] + point_coords[1])/2, -0.1, r"($\zeta$, $\zeta$, 0)", fontsize=12, ha="center", va="top")
ax.text((point_coords[3] + point_coords[2])/2, -0.1, r"($\zeta$, $\zeta$, $\zeta$)", fontsize=12, ha="center", va="top")


handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(
    zip(handles, labels)) if l not in labels[:i]]

z = zip(*unique)

handles, labels = z

legax.legend(handles, labels, fontsize=12)

plt.tight_layout()
fig.savefig("..Test_Plots/Phonons.png")