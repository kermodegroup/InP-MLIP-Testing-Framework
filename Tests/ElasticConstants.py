from active_model import *
from ase.io import read
import numpy as np
from matscipy.elasticity import fit_elastic_constants, elastic_moduli
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGSLineSearch, BFGS
from ase.constraints import ExpCellFilter
import matplotlib.pyplot as plt
import os
from ase.units import GPa
from Utils.jsondata import add_info

def test_calc(calc, calc_name):
    ats = read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")
    Cs = np.zeros((6, 6))
    C_errs = np.zeros_like(Cs)

    at = ats.copy()
    at.calc = calc

    # Allow only diagonal cell parameters to relax
    filter = ExpCellFilter(at, mask=[True]*3 + [False]*3)

    opt = PreconLBFGS(filter)

    opt.run(fmax=1e-3, smax=1e-3)

    alat = np.average(np.diag(at.cell[:, :]))


    # Parameters for the elastic constant search
    # delta_max is the maximum strain in each direction
    delta_max = 2E-3
    # nsteps is the number of images used to fit the stress-strain curve
    nsteps = 15

    Cs, C_errs = fit_elastic_constants(
        at, N_steps=nsteps, delta=2*delta_max/nsteps, optimizer=BFGSLineSearch, fmax=1e-3, graphics=True)

    plt.savefig("../Test_Plots" + os.sep + calc_name + os.sep + "C_fit.png")
    
    # C11, C12, C44 in GPa
    c = np.array([Cs[0, 0], Cs[0, 1], Cs[-1, -1]]) / GPa
    c_err = np.array([C_errs[0, 0], C_errs[0, 1], C_errs[3, 3]]) / GPa

    dat = np.array([c, c_err])

    # Fit elastic moduli
    E, nu, Gm, B, K = elastic_moduli(Cs)

    return alat, c, c_err, np.average(E)/GPa, nu, Gm, B, K

calc_name = active_model_name
calc = get_model(active_model_name)


alat, Cs, C_errs, E, nu, _, _, _= test_calc(calc, calc_name)



calc_data = {
    "C11" : Cs[0],
    "C12" : Cs[1],
    "C44" : Cs[2]
}

add_info(active_model_name, calc_data)

print(calc_name)
print("alat: ", alat)
print("C_11: ", Cs[0], " +- ", C_errs[0])
print("C_12: ", Cs[1], " +- ", C_errs[1])
print("C_44: ", Cs[2], " +- ", C_errs[2])
print("E: ", E)
print("nu: ", nu)