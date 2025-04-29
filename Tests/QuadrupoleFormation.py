from matscipy.dislocation import Quadrupole, DiamondGlide30degreePartial, DiamondGlide90degreePartial, DiamondGlide60Degree, DiamondGlideScrew
from Utils.utils import get_bulk
from ase.optimize.precon import PreconLBFGS
from ase.io import read, write
import numpy as np
from Utils.jsondata import add_info
from active_model import *
from matscipy.neighbours import mic

calc_name = active_model_name
calc = get_model(active_model_name)

disloc_names = ["30 degree Partial", "90 degree Partial"]

dft_structs = [read("DFT_Reference/Quadrupoles/DFT_Quads_0/DFT_Quads_0.geom", index="-1"), read("DFT_Reference/Quadrupoles/DFT_Quads_1/DFT_Quads_1.geom", index="-1")]

structs = []
calc_data = {}

dft_a0 = 5.901273599999999


for k, disloc in enumerate([DiamondGlide30degreePartial, DiamondGlide90degreePartial]):
    ref_bulk, C11, C12, C44 = get_bulk(calc, elast=True)
    d = Quadrupole(disloc, ref_bulk, C11, C12, C44)
    struct = dft_structs[k].copy()
    cell = struct.cell[:, :].copy() / dft_a0 * ref_bulk.cell[0, 0]
    struct.set_cell(cell, scale_atoms=True)

    ref_bulk.calc = calc
    E_bulk = ref_bulk.get_potential_energy() / len(ref_bulk) * len(struct)
    struct.calc = calc
    opt = PreconLBFGS(struct)
    opt.run(1e-3, steps=400)

    if opt.converged():
        
        E_form = struct.get_potential_energy() - E_bulk

        dft_rs = dft_structs[k].get_positions() / dft_a0
        rs = struct.get_positions() / ref_bulk.cell[0, 0]

        errs = mic(rs - dft_rs, np.eye(3))

        dr = np.sqrt(np.average(np.linalg.norm(errs, axis=-1)**2))

        calc_data[disloc_names[k] + " Eform"] = E_form
        calc_data[disloc_names[k] + " dr"] = dr * 100
    else:
        calc_data[disloc_names[k] + " Eform"] = None
        calc_data[disloc_names[k] + " dr"] = None

    struct.arrays["x_err"] = errs[:, 0]
    struct.arrays["y_err"] = errs[:, 1]
    struct.arrays["z_err"] = errs[:, 2]
    struct.arrays["total_err"] = np.linalg.norm(errs, axis=-1)
    structs.append(struct)

write(f"../Test_Results/{calc_name}/Quadrupole_structs.xyz", structs)

add_info(calc_name, calc_data)