from Utils.utils import get_bulk
from ase.io import read, write
from Utils.jsondata import add_info
from active_model import *
from ase.mep.neb import NEB, NEBOptimizer
from ase.optimize import BFGSLineSearch
from ase.calculators.singlepoint import SinglePointCalculator

def run_neb(name, calc, calc_name, nims, ftol=1e-4, neb_ftol=5e-3, steps=500, neb_steps=500, climb=True, interpolate=False):
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

        xyz_fname = "../Test_Results/" + calc_name + os.sep + name + "_Quadrupole_Migration_structs.xyz"
        
        
        if os.path.exists(xyz_fname):
            images = read(xyz_fname, index=":")
        else:
            images = read(f"Misc_Reference/Quadrupole_Migration/" + name + "_Quadrupole_Migration_structs.xyz", index=":")
            start = images[0]
            end = images[-1]

            #images = do_neb(images, climb=False, interpolate=True)

        images = do_neb(images, climb=True, interpolate=False)

        for image in images:
            image.calc = calc
            E = image.get_potential_energy()
            image.calc = SinglePointCalculator(image, energy=E)

        write(xyz_fname, images)
        return images 


calc_name = active_model_name
calc = get_model(active_model_name)

short_names = ["30deg", "90deg"]

nims=15

for name in short_names:
    run_neb(name, calc, calc_name, nims)