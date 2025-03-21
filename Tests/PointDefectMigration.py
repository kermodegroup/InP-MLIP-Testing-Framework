from StructGen import interstitial as make_interstitial
from file_io import read, write, ase_read
import numpy as np
from models import *
from ase.optimize import BFGSLineSearch
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from neb_core import do_NEB
import os
from active_model import *
from ase.units import kB
from itertools import combinations

supercell_size = 2
zb = read("Accurate_Bulk/ZB_Bulk.xyz", index="-1")

base_cell = zb.cell

zb = zb * (supercell_size, supercell_size, supercell_size)

tol = 1E-4
neb_ftol = 1e-3
paths = os.listdir("PD_Migration_Paths")
interstitials = [item for item in paths if "Interstitial" in item]
vacancies = [item for item in paths if "Vacancy" in item]

def eval(calc, calc_name, write_traj=False, species = ["In", "P"], interstitial=True, vacancy=True, nims=9, **kwargs):
    def run_neb(name):
        
        if name != "P_Vacancy_Migration_5->3":
            return

        traj_fname = None
        xyz_fname = "../Saved_Data/" + calc_name + os.sep + "PDMigration/" + name + "_final_ims.xyz"
        
        
        if os.path.exists(xyz_fname):
            images = ase_read(xyz_fname, index=":")
        else:
            images = []


        if len(images) == nims:
            start = images
            end = images[-1]
            skip_non_ci = False
        else:
            images = ase_read(f"PD_Migration_Paths/" + vac, index=":")
            start = images[0].copy()
            end = images[-1].copy()
            skip_non_ci = False


            for ats in [start, end]:
                ats.calc = calc
                opt = BFGSLineSearch(ats)
                opt.run(1e-3, steps=500)            

                if not opt.converged():
                    return


        neb, e = do_NEB(calc, start, end, traj_fname, nims=nims, skip_non_ci=skip_non_ci, **kwargs)
        print(e)
        if e is None:
            write(xyz_fname, neb.images)
    
    # Interstitial
    if interstitial:
        for i, inter in enumerate(interstitials):
            name = inter[:-4]

            print(name)

            run_neb(name)
    
    # Vacancy          
    if vacancy:
        for i, vac in enumerate(vacancies):
            name = vac[:-4]

            print(name)

            run_neb(name)


calc_name = active_model_name
calc = get_model(active_model_name)

dump_dir = "../Saved_Data/" + calc_name + os.sep + "PDMigration"

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

eval(calc, calc_name, write_traj=False, nims=21, neb_ftol=3e-2, max_nsteps=500, ci=False, refine=False,
     species=["In", "P"], interstitial=True, vacancy=True, ftol=1e-3, apply_constraints=False, method="ode")