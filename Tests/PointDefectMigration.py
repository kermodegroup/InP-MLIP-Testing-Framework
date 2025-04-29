from ase.io import read, write
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGSLineSearch
from ase.mep.neb import NEB, NEBOptimizer
import os
from ase.calculators.singlepoint import SinglePointCalculator
from active_model import *

supercell_size = 2
zb = read("DFT_Reference/Bulk/ZB_Bulk.xyz", index="-1")

base_cell = zb.cell

zb = zb * (supercell_size, supercell_size, supercell_size)

paths = os.listdir("Misc_Reference/PD_Migration_Paths")
interstitials = [item for item in paths if "Interstitial" in item]
vacancies = [item for item in paths if "Vacancy" in item]

def eval(calc, calc_name, species = ["In", "P"], interstitial=True, vacancy=True, nims=15, steps=500, neb_steps=500,
            ftol=1E-3, neb_ftol = 1e-2):
    def run_neb(name, climb=True, interpolate=False):

        def do_neb(images, climb=True, interpolate=False):
            start = images[0]
            end = images[-1]

            if interpolate:
                images = [start.copy() for i in range(nims-1)] + [end.copy()]
                neb = NEB(images)
                neb.interpolate(method="idpp", mic=True)
                images = neb.images
            
            for ats in [start, end]:
                ats.calc = calc
                opt = BFGSLineSearch(ats)
                opt.run(ftol, steps=steps)      

            for image in images:
                image.calc = calc

            neb = NEB(images, allow_shared_calculator=True, climb=climb)
            opt = NEBOptimizer(neb)
            opt.run(neb_ftol, steps=neb_steps)

            return neb.images

        xyz_fname = "../Test_Results/" + calc_name + os.sep + "PDMigration/" + name + ".xyz"
        
        
        if os.path.exists(xyz_fname):
            images = read(xyz_fname, index=":")
        else:
            images = read(f"Misc_Reference/PD_Migration_Paths/" + name + ".xyz", index=":")
            start = images[0]
            end = images[-1]

            #images = do_neb(images, climb=False, interpolate=True)

        images = do_neb(images, climb=True, interpolate=False)

        for image in images:
            image.calc = calc
            E = image.get_potential_energy()
            image.calc = SinglePointCalculator(image, energy=E)

        write(xyz_fname, images)   


        
    
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

dump_dir = "../Test_Results/" + calc_name + os.sep + "PDMigration"

if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

eval(calc, calc_name)