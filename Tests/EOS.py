import os
from active_model import *
import matplotlib.pyplot as plt
import numpy as np
from Utils.file_io import read
from tqdm import tqdm, trange
from ase.optimize import BFGSLineSearch
from Utils.jsondata import add_info
import json

def _get_dft_data():
    # ZB
    zb_structs = read("EOS/ZB_EOS.xyz", index=":") + [read("Bulk/ZB_Bulk.xyz", index="-1")]

    e_zb = np.array([ats.get_potential_energy()/len(ats)
                    for ats in zb_structs])
    v_zb = np.array([ats.get_volume()/len(ats) for ats in zb_structs])

    # WZ

    wz_structs = read("EOS/WZ_EOS.xyz", index=":") + [read("Bulk/WZ_Bulk.xyz", index="-1")]

    e_wz = np.array([ats.get_potential_energy()/len(ats)
                    for ats in wz_structs])
    v_wz = np.array([ats.get_volume()/len(ats) for ats in wz_structs])

    zb_min = np.min(e_zb)
    wz_min = np.min(e_wz)
    e_min = np.min([zb_min, wz_min])

    e_zb -= e_min
    e_wz -= e_min

    return [v_zb, v_wz], [e_zb, e_wz]


def test_calc(calc, vol_lims=[8.0, 45.0], npoints=5000):
    '''
    Do EOS test on calc

    '''

    zb = read("Accurate_Bulk/ZB_Bulk.xyz", index="-1")

    wz = read("Accurate_Bulk/WZ_Bulk.xyz", index="-1")

    v = []
    e = []

    for ats in tqdm([zb, wz]):
        opt = BFGSLineSearch(ats)
        opt.run(1E-3)

        at_vol = ats.get_volume() / len(ats)
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
    e_min = min([np.min(es) for es in e])
    e = [es - e_min for es in e]
    return v, e


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

v, e = test_calc(calc, npoints=2000)

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

with open(f"../Test_Results/{active_model_name}/EOS.json", "w") as f:
    json.dump(d, f, indent=4)


v_zb = np.array(v[0])
mask = (v_zb > 23.0) * (v_zb < 27.0)
v_zb = v_zb[mask]
e_zb = np.array(e[0])[mask]

calc_zb_V0 = v_zb[np.argmin(e_zb)]
calc_zb_a0 = (8 * calc_zb_V0)**(1/3)


calc_data = {
    "V0" : calc_zb_V0,
    "a0" : calc_zb_a0
}

add_info(active_model_name, calc_data)