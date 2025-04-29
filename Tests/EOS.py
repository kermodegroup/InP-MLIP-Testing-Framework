import os
from active_model import *
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from tqdm import tqdm, trange
from ase.optimize import BFGSLineSearch
from ase.optimize.precon import PreconLBFGS
from Utils.jsondata import add_info
import json
from ase.constraints import ExpCellFilter 

zoom_lims = [23.0, 27.0]
zoom_mask = lambda v: (v > 23.0) * (v < 27.0)

def _get_dft_data():
    # ZB
    zb_structs = read("DFT_Reference/EOS/ZB_EOS.xyz", index=":") + [read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")]

    e_zb = np.array([ats.get_potential_energy()/len(ats)
                    for ats in zb_structs])
    v_zb = np.array([ats.get_volume()/len(ats) for ats in zb_structs])

    # WZ

    wz_structs = read("DFT_Reference/EOS/WZ_EOS.xyz", index=":") + [read("DFT_Reference/Bulk/WZ_Bulk.xyz", index="-1")]

    e_wz = np.array([ats.get_potential_energy()/len(ats)
                    for ats in wz_structs])
    v_wz = np.array([ats.get_volume()/len(ats) for ats in wz_structs])

    zb_min = np.min(e_zb)
    wz_min = np.min(e_wz)
    e_min = np.min([zb_min, wz_min])

    e_zb -= e_min
    e_wz -= e_min

    return [v_zb, v_wz], [e_zb, e_wz]


def test_calc(calc, vol_lims=[13.0, 31.0], npoints=5000):
    '''
    Do EOS test on calc

    '''

    zb = read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")

    wz = read("DFT_Reference/Bulk/WZ_Bulk.xyz", index="-1")

    v = []
    e = []

    a0s = []
    V0s = []

    for ats in tqdm([wz, zb]):
        ats.calc = calc
        
        filter = ExpCellFilter(ats, mask=[True]*3 + [False]*3)

        opt = PreconLBFGS(filter)

        opt.run(fmax=1e-4, smax=1e-6)

        at_vol = ats.get_volume() / len(ats)
        V0s.append(at_vol)
        a0s.append((8 * at_vol)**(1/3))
        cell = ats.cell

        eps = np.linspace((vol_lims[0]/at_vol)**(1/3),
                          (vol_lims[1]/at_vol)**(1/3), npoints)

        vols = np.zeros_like(eps)
        energies = np.zeros_like(vols)

        for i in trange(npoints, leave=False):
            ep = eps[i]
            at = ats.copy()
            at.set_cell(cell * ep, scale_atoms=True)
            at.calc = calc
            energies[i] = at.get_potential_energy()
            vols[i] = at.get_volume()

        vols /= len(ats)
        energies /= len(ats)

        e.append(energies)
        v.append(vols)
        e_min = np.min(energies[zoom_mask(vols)])

    # Subtract off the energy corresponding to the ZB a_0 structure
    e = [es - e_min for es in e]
    return v, e, V0s, a0s


v, e = _get_dft_data()

v_zb, v_wz = v
e_zb, e_wz = e

idx_zb = np.argsort(v_zb)
idx_wz = np.argsort(v_wz)

v_zb = v_zb[idx_zb]
e_zb = e_zb[idx_zb]
v_wz = v_wz[idx_wz]
e_wz = e_wz[idx_wz]
 
d = {
    "V_ZB" : list(v_zb),
    "E_ZB" : list(e_zb),
    "V_WZ" : list(v_wz),
    "E_WZ" : list(e_wz)
}

with open("../Test_Results/DFT/EOS.json", "w") as f:
    json.dump(d, f, indent=4)

dft_zb_V0 = v_zb[np.argmin(e_zb)]
dft_zb_a0 = (8 * dft_zb_V0)**(1/3)


dft_data = {
    "V0" : dft_zb_V0,
    "a0" : dft_zb_a0
}

add_info("DFT", dft_data)

calc_name = active_model_name
calc = get_model(active_model_name)

v, e, v0s, a0s = test_calc(calc, npoints=2000)

v_wz, v_zb = v
e_wz, e_zb = e

idx_zb = np.argsort(v_zb)
idx_wz = np.argsort(v_wz)

v_zb = v_zb[idx_zb]
e_zb = e_zb[idx_zb]
v_wz = v_wz[idx_wz]
e_wz = e_wz[idx_wz]
 
d = {
    "V_ZB" : list(v_zb),
    "E_ZB" : list(e_zb),
    "V_WZ" : list(v_wz),
    "E_WZ" : list(e_wz)
}

with open(f"../Test_Results/{active_model_name}/EOS.json", "w") as f:
    json.dump(d, f, indent=4)


calc_zb_V0 = v0s[1]
calc_zb_a0 = a0s[1]


calc_data = {
    "V0" : v0s[1],
    "a0" : a0s[1]
}

add_info(active_model_name, calc_data)