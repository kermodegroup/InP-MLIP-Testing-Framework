import matplotlib.pyplot as plt
from matscipy.utils import get_structure_types
#from ase.visualize.plot import plot_atoms
import numpy as np
import matplotlib.patches as mpatches
from matscipy.neighbours import neighbour_list



def plot_atoms(ax, atoms, colours, labels, In_rad=0.8, P_rad=0.6, r_bond=3.0, w_bond=0.4, plot_bonds=True, x_lims=None, y_lims=None, supercell=None, lw=0.1):
    
    #atoms.wrap()

    cell = atoms.cell[:, :].copy()

    x_cell = np.array([
        [0, 0],
        cell[0, :2],
        cell[0, :2] + cell[1, :2],
        cell[1, :2]
    ])

    if supercell is not None:
        atoms = atoms.copy()
        atoms = atoms * supercell
        colours = colours * np.sum(supercell)
        labels = labels * np.sum(supercell)

    p = atoms.positions


    if x_lims is None:
        x_lims = [np.min([np.min(x_cell[:, 0]), np.min(p[:, 0])]), 
                np.max([np.max(x_cell[:, 0]), np.max(p[:, 0])])]
    if y_lims is None:
        y_lims = [np.min([np.min(x_cell[:, 1]), np.min(p[:, 1])]), 
                np.max([np.max(x_cell[:, 1]), np.max(p[:, 1])])]

    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    
    radii = [In_rad if spec == "In" else P_rad for spec in atoms.get_chemical_symbols()]
    p = atoms.positions[:, :2]


    patches = [
        mpatches.Polygon(x_cell, color="k", fill=False)
    ]

    if plot_bonds:
        neigh_ats = atoms.copy()
        #neigh_ats.pbc = [False, False, False]
        ibond, jbond, Dbond = neighbour_list("ijD", neigh_ats, float(r_bond))
        
        for k in range(len(ibond)):
            idx = ibond[k]
            jdx = jbond[k]
            D = Dbond[k, :2] / 2

            D_perp = np.array([D[1], -D[0]])
            D_perp /= np.linalg.norm(D_perp)

            rect1_coords = [
                p[idx, :] - 0.5 * w_bond * D_perp,
                p[idx, :] - 0.5 * w_bond * D_perp + D,
                p[idx, :] + 0.5 * w_bond * D_perp + D,
                p[idx, :] + 0.5 * w_bond * D_perp,
            ]

            rect2_coords = [
                p[jdx, :] - 0.5 * w_bond * D_perp,
                p[jdx, :] - 0.5 * w_bond * D_perp - D,
                p[jdx, :] + 0.5 * w_bond * D_perp - D,
                p[jdx, :] + 0.5 * w_bond * D_perp,
            ]

            patches.extend([
                mpatches.Polygon(rect1_coords, ec="k", fc=colours[idx], lw=lw),
                mpatches.Polygon(rect2_coords, ec="k", fc=colours[jdx], lw=lw)
            ])


    for i in range(len(atoms)):
        patches.append(mpatches.Circle(p[i, :], radius=radii[i], ec="k", fc=colours[i], lw=lw))


    for patch in patches:
        ax.add_patch(patch)