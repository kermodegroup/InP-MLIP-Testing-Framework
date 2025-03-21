from models import GAP_committee
from active_model import *
from file_io import read
import numpy as np
from matscipy.elasticity import fit_elastic_constants, elastic_moduli
from models import GAP, ACE
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGSLineSearch, BFGS
import os
from ase.io import read as ase_read
import json
from ase.constraints import ExpCellFilter
from jsondata import add_info

data_dir = "../Saved_Data"

fmax = 1E-4

data = {}

In_iso = read("IsolatedAtom/Iso_In.xyz", index="-1")
P_iso = read("IsolatedAtom/Iso_P.xyz", index="-1")


def get_binding_energy(struct, E_iso_In, E_iso_P, calc):
    spec = np.array(struct.get_chemical_symbols())

    In_nats = sum(spec == "In")
    P_nats = sum(spec == "P")

    if calc is not None:
        struct.calc = calc

    E_bind = struct.get_potential_energy() - In_nats * E_iso_In - P_nats * E_iso_P

    return E_bind


def test_calc(calc, calc_name, err_func=None):

    data = {}

    with open(f"../Saved_Data/PointDefects/{calc_name}_ChemPot.json") as f:
        chempot = json.load(f)

    os.makedirs(f"../Saved_Data/{calc_name}/PointDefectTraj", exist_ok=True)
    files = os.listdir("PointDefectStructs")

    formation_energies = {"zb_E0":{}, "wz_E0":{}}

    if type(calc) != list:  # Single model, not list
        calc = [calc]

    for file in files:
        name = file.split(".")[0]
        formation_energies[name] = {"raw_energies":{}}

    for i in range(len(calc)):
        sub_calc = calc[i]

        if sub_calc is not None:
            In_iso.calc = sub_calc
        E_iso_In = In_iso.get_potential_energy()


        if sub_calc is not None:
            P_iso.calc = sub_calc
        E_iso_P = P_iso.get_potential_energy()


        E_iso = 0.5 * (E_iso_In + E_iso_P)

        zb_bulk = read("Accurate_Bulk/ZB_Bulk.xyz", index="-1")
        wz_bulk = read("Accurate_Bulk/WZ_Bulk.xyz", index="-1")
        
        zb_bulk.calc = sub_calc
        filter = ExpCellFilter(zb_bulk, mask=[True]*6)

        opt = PreconLBFGS(filter)

        opt.run(fmax=1e-4, smax=1e-4)

        wz_bulk.calc = sub_calc
        filter = ExpCellFilter(wz_bulk, mask=[True]*6)

        opt = PreconLBFGS(filter)

        opt.run(fmax=1e-6, smax=1e-6)

        spec = np.array(zb_bulk.get_chemical_symbols())
        zb_bulk_n_In = np.sum(spec == "In") * 8
        zb_bulk_n_P = np.sum(spec == "P") * 8


        spec = np.array(wz_bulk.get_chemical_symbols())
        wz_bulk_n_In = np.sum(spec == "In") * 8
        wz_bulk_n_P = np.sum(spec == "P") * 8

        bulk_n_In = {"zb":zb_bulk_n_In, "wz": wz_bulk_n_In}
        bulk_n_P = {"zb":zb_bulk_n_P, "wz": wz_bulk_n_P}

        bulk_n = {"zb":len(zb_bulk), "wz":len(wz_bulk)}

        zb_E0 = zb_bulk.get_potential_energy() * 8

        zb_cell = zb_bulk.cell[:, :] * 2


        wz_E0 = wz_bulk.get_potential_energy() * 8

        wz_cell = wz_bulk.cell[:, :] * 2

        formation_energies["zb_E0"][i] = zb_E0
        formation_energies["wz_E0"][i] = wz_E0

        spec = np.array(zb_bulk.get_chemical_symbols())

        formation_energies["zb_E0"]["n_In"] = int(np.sum(spec=="In"))
        formation_energies["zb_E0"]["n_P"] = int(np.sum(spec=="P"))
        
        spec = np.array(wz_bulk.get_chemical_symbols())

        formation_energies["wz_E0"]["n_In"] = int(np.sum(spec=="In"))
        formation_energies["wz_E0"]["n_P"] = int(np.sum(spec=="P"))

        if sub_calc is None:
            idx = "-1"
        else:
            idx = "-1"

        for file in files:
            name = file.split(".")[0]
            ats = ase_read("PointDefectStructs/" + file, index=idx)


            crystal_struct = "zb" if "ZB" in name else "wz"

            if crystal_struct == "zb":
                cell = zb_cell
            else:
                cell = wz_cell

            spec = np.array(ats.get_chemical_symbols())

            n = len(ats)

            dn = n - bulk_n[crystal_struct] * 8

            n_In = np.sum(spec == "In")
            n_P = np.sum(spec == "P")

            d_n_In = n_In - bulk_n_In[crystal_struct]
            d_n_P = n_P - bulk_n_P[crystal_struct]

            dE = d_n_In * E_iso_In + d_n_P * E_iso_P


            if sub_calc is not None:
                ats.set_cell(cell, scale_atoms=True)
                ats.calc = sub_calc
                opt = BFGSLineSearch(ats, trajectory=f"../Saved_Data/{calc_name}/PointDefectTraj/{file[:-4]}.traj")
                opt.run(fmax)

            E_bind = ats.get_potential_energy()

            formation_energies[name]["raw_energies"][i] = E_bind

            if crystal_struct == "zb":
                E0 = zb_E0
                formation_energies[name][i] = E_bind - zb_E0
            else:
                E0 = wz_E0
                formation_energies[name][i] = E_bind - wz_E0

            formation_energies[name][i] -= dE

            
            spec = np.array(ats.get_chemical_symbols())

            formation_energies[name]["n_In"] = int(np.sum(spec=="In"))
            formation_energies[name]["n_P"] = int(np.sum(spec=="P"))

    for file in files:
        name = file.split(".")[0]
        average = np.average(
            np.array([formation_energies[name][i] for i in range(len(calc))]))
        std = np.std(np.array([formation_energies[name][i] for i in range(len(calc))]))
        formation_energies[name]["Average"] = average
        formation_energies[name]["Error"] = std

        data[name] = average
    with open(data_dir + os.sep + "PointDefects" + os.sep + calc_name + "_Formation_Energies.json", "w") as f:
        json.dump(formation_energies, f, indent=4)

    add_info(calc_name, data)


active_model = get_model(active_model_name)

test_calc(active_model, active_model_name)
