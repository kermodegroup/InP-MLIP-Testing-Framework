import numpy as np
from ase.io import read
from ase.build import bulk
import ase
from ase.constraints import ExpCellFilter
from ase.optimize.precon import PreconLBFGS
from ase.units import GPa
from matscipy.elasticity import fit_elastic_constants
from ase.calculators.singlepoint import SinglePointCalculator

def rattle_structs(structs, n_rattles=100, r_tol=1e-1):
    rng = np.random.RandomState()

    if type(structs) == ase.Atoms:
        structs = [structs]

    ims = []
    for image in structs:
        for i in range(n_rattles):
            ats = image.copy()
            ats.rattle(r_tol, rng=rng)
            ims.append(ats)

    return ims

def get_bulk(calc = None, elast=False, nsteps=5, ftol=1e-6, stol=1e-6):
    ats = read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")
    alat = ats.cell[0, 0]

    #ats = bulk("InP", "zincblende", alat, cubic=True)

    if calc is not None:
        ats.calc = calc

        filter = ExpCellFilter(ats, mask=[True]*3 + [False]*3)

        opt = PreconLBFGS(filter)

        opt.run(fmax=ftol, smax=stol)

        alat = ats.cell[0, 0]
        new_ats = bulk("InP", "zincblende", alat, cubic=True)

        if elast:
            delta_max = 2E-3
            Cs, C_errs = fit_elastic_constants(
                ats, N_steps=nsteps, delta=2*delta_max/nsteps, optimizer=PreconLBFGS, fmax=ftol, graphics=False,
                verbose=False)
            c = np.array([Cs[0, 0], Cs[0, 1], Cs[-1, -1]]) / GPa
            return new_ats, *c
    
    alat = ats.cell[0, 0]
    new_ats = bulk("InP", "zincblende", alat, cubic=True)
    
    if elast:
        # return rough DFT elastic constants
        return new_ats, 97, 53, 47
    else:
        return new_ats

def get_slice_mask(struct, eps=1e-3):
    ats = struct.copy()
    ats.positions[:, 2] += eps
    ats.wrap()
    mask = ats.get_scaled_positions()[:, 2] <= 0.5

    return mask

def slice_disloc(struct, mask=None, return_mask=False, eps=1e-3):
    if mask is None:
        mask = get_slice_mask(struct, eps)

    ats = struct.copy()
    del ats[mask]
    ats.cell[2, 2] /= 2
    ats.wrap()

    if return_mask:
        return ats, mask
    else:
        return ats

def apply_singlepoint(ats):
    E = ats.get_potential_energy()
    F = ats.get_forces()

    scalc = SinglePointCalculator(ats, energy=E, forces=F)
    ats.calc = scalc