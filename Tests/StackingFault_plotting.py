from ase.io import read
from matscipy.gamma_surface import StackingFault
import matplotlib.pyplot as plt
from ase.units import _e
import numpy as np
from Utils.jsondata import add_info
from active_model import *
from Utils.plot_atoms import plot_atoms
from matscipy.utils import get_structure_types
import numpy as np
from ase.build import rotate
import matplotlib.gridspec as gridspec

nims = 81

def vtm(dir, brackets):
    repr = " "
    for item in dir:
        if item < 0:
            repr += r"\widebar{" + str(-item) + r"} "
        else:
            repr += str(item) + " "
    return r"\left" + brackets[0] + repr + r"\right" + brackets[1]
planes = [(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
dirs = [(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 1, -2)]
zreps = [3, 6, 6, 2]
max_lims = [1, 1, 1, 1]

bulk = read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")

calc_names = [model for model in plot_models if model != "DFT"]
dft_data = {}


si = True

if si:
    mul = _e * 1e20
    units = r"J/m$^2$"
else:
    mul = 1
    units = r"eV/$ \mathrm{A}^2$"

ns = [9, 9, 9, 9]


fig, ax = plt.subplots(figsize=(12, 13))
ax.axis("off")

topleftgrid = gridspec.GridSpec(1, 3)
toprightgrid = gridspec.GridSpec(1, 3)
bottomleftgrid = gridspec.GridSpec(1, 3)
bottomrightgrid = gridspec.GridSpec(1, 3)

midgrid = gridspec.GridSpec(2, 2)

bottomleftgrid.update(left=0.08, right=0.475, bottom=0.02, top=0.18)
bottomrightgrid.update(left=0.56, right=0.95, bottom=0.02, top=0.18)
midgrid.update(left=0.08, right=0.95, bottom=0.23, top=0.79, hspace=0.4)
topleftgrid.update(left=0.08, right=0.475, bottom=0.82, top=0.98)
toprightgrid.update(left=0.56, right=0.95, bottom=0.82, top=0.98)


topax = [plt.subplot(topleftgrid[i]) for i in range(3)] + [plt.subplot(toprightgrid[i]) for i in range(3)]
botax = [plt.subplot(bottomleftgrid[i]) for i in range(3)] + [plt.subplot(bottomrightgrid[i]) for i in range(3)]

ax = [
    plt.subplot(midgrid[0, 0]), 
    plt.subplot(midgrid[0, 1]),
    plt.subplot(midgrid[1, 0]),
    plt.subplot(midgrid[1, 1])
]


for i in range(6):
    topax[i].set_xticks([])
    topax[i].set_yticks([])
    botax[i].set_xticks([])
    botax[i].set_yticks([])

atoms_axes = [
    topax[:3],
    topax[3:],
    botax[:3],
    botax[3:]
]

plot_at_centers = [
    4, 4, 4, 4
]

ylims = [
    [8.851910493406287, 26.55573148021886],
    [28.32611357890012, 42.48917036835017],
    [20.02958638508013, 30.04437957762019],
    [10.221305492386234, 30.6639164771587]
]


for i in range(len(planes)):
    plane = planes[i]
    dir = dirs[i]
    zr = zreps[i]
    yl = max_lims[i]

    fault = StackingFault(bulk, plane, dir)
    fault.generate_images(nims, z_reps=zr, path_ylims=[0, yl/2])

    ax[i].tick_params(axis='both', which='major', labelsize=12)

    for idx in range(len(calc_names)):
        active_model_name = calc_names[idx]
        ims = read(f"../Test_Results/{active_model_name}/StackingFaultStructs_{plane}_{dir}.xyz", index=":")

        Es = []
        for image in ims:
            cell = image.cell[:, :]
            surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
            Es.append(image.get_potential_energy() / surface_area)
        Es = np.array(Es)
        Es -= Es[0]
        xs = [image.cell[2, 1] for image in ims]

        ax[i].plot(xs, np.array(Es) * mul, label=active_model_name)
    
    
    p_text = "_".join([str(idx) for idx in plane])
    d_text = "_".join([str(idx) for idx in dir])

    pt = "".join([str(idx) for idx in plane])
    dt = "".join([str(idx) for idx in dir])

    dft_ims = [read(f"DFT_Reference/Stacking_Faults/StackingFaultStructs_{pt}_{dt}_{idx}/StackingFaultStructs_{pt}_{dt}_{idx}.geom", index="-1") for idx in range(ns[i])]
    ref_nats = len(dft_ims[0])

    dft_ims = dft_ims
    dft_Es = []
    for image in dft_ims:
        cell = image.cell[:, :]
        surface_area = np.linalg.norm(np.cross(cell[0, :], cell[1, :]))
        dft_Es.append(image.get_potential_energy() / surface_area)
    dft_Es = np.array(dft_Es)
    dft_Es -= dft_Es[0]

    dft_data[f"E_{plane}_{dir}"] = np.max(dft_Es) * _e * 1e20

    if i == 3:
        dft_data["Disloc_SF_Form"] = (dft_Es[-1] - dft_Es[0]) * _e * 1e20 * 1000

    xs = [image.cell[2, 1] for image in ims]

    ys = [image.cell[2, 1] for image in dft_ims]

    ax[i].scatter(ys, np.array(dft_Es) * mul, label="RSCAN DFT", color="k", marker="x")
    
    if i == 1:
        ax[i].legend(fontsize=13)

    ax[i].set_ylabel(f"Energy Density ({units})", fontsize=13)
    ax[i].set_xlabel(f"${vtm(np.array(dir), brackets='[]')}$ Displacment (Å)", fontsize=13)
    ax[i].set_title(f"${vtm(np.array(plane), brackets='()')}{vtm(np.array(dir), brackets='[]')}$ Stacking Fault", fontsize=13)

    plot_ats = [dft_ims[0], dft_ims[plot_at_centers[i]], dft_ims[-1]]
    plot_ats_xs = [0, ys[plot_at_centers[i]], xs[-2]]


    for idx in range(3):
        cell = plot_ats[idx].cell[:, :]
        #ylims = [0.5 * cell[2, 2] , 1.5 * cell[2, 2]]
        ats = plot_ats[idx].copy()
        p = ats.positions
        
        rotate(ats, (0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0))

        xlims = [-1.5, 2 * cell[1, 1] + .5]
        s = xlims[1] - xlims[0]
        aspect = 1.2
        ylims = [cell[2, 2] - aspect * s / 2, cell[2, 2] + aspect * s / 2]
        cell = ats.cell[:, :]
        
        new_cell = np.array([
            cell[1, :],
            cell[2, :],
            cell[0, :],
        ])

        if i == 1:
            supercell = (2, 2, 1)
        else:
            supercell = (1, 2, 1)

        ats.cell = new_cell
        atom_labels, struct_names, colors = get_structure_types(ats, diamond_structure=True)
        atom_colors = [colors[atom_label] for atom_label in atom_labels]
        plot_atoms(atoms_axes[i][idx], ats, atom_colors, atom_labels, x_lims=xlims, y_lims=ylims, supercell=supercell)
        atoms_axes[i][idx].set_aspect("equal")
        atoms_axes[i][idx].text(0.5, 1.0, f"{plot_ats_xs[idx]:.1f} Å ", color="k", va="bottom", ha="center", transform=atoms_axes[i][idx].transAxes, fontsize=15)

        atoms_axes[i][idx].set_xlabel(f"${vtm(np.array(dir), brackets='[]')}$", ha="center")
        atoms_axes[i][idx].set_ylabel(f"${vtm(np.array(plane), brackets='[]')}$", rotation=90)

if si:
    plt.savefig(f"../Test_Plots/StackingFaultPlot_SI.png", dpi=200)
else:
    plt.savefig(f"../Test_Plots/StackingFaultPlot.png", dpi=200)


add_info("DFT", dft_data)