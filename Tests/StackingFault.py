from active_model import get_model, active_model_name
from ase.io import read, write
from matscipy.dislocation import DiamondGlideScrew
from matscipy.gamma_surface import StackingFault
import matplotlib.pyplot as plt
import os
from ase.constraints import ExpCellFilter
#from ase.neb import NEB, NEBOptimizer
from ase.units import _e
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from Utils.jsondata import add_info
from Utils.utils import get_bulk
nims = 81

planes = [(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
dirs = [(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 1, -2)]
zreps = [3, 6, 6, 2]
max_lims = [1, 1, 1, 1/3]


active_model = get_model(active_model_name)

bulk = get_bulk(active_model, elast=False)

data = {}

print(active_model_name)

for i in range(len(planes)):
    plane = planes[i]
    dir = dirs[i]
    zr = zreps[i]
    yl = max_lims[i]

    save_file = f"../Test_Results/{active_model_name}/StackingFaultStructs_{plane}_{dir}.xyz"
    
    fault = StackingFault(bulk, plane, dir)
    fault.generate_images(nims, z_reps=zr, path_ylims=[0, yl])

    print(plane, dir, len(fault.images[0]))
    if os.path.exists(save_file):
        ats = read(save_file, index=":")
        if len(ats) == nims:
            fault.images = ats

    Es = fault.get_energy_densities(active_model, relax=True, cell_relax=False, ftol=1e-4)

    for image in fault.images:
        image.calc = active_model
        E = image.get_potential_energy()
        F = image.get_forces()
        image.calc = SinglePointCalculator(image, energy=E, forces=F)

    write(save_file, fault.images)

    data[f"E_{plane}_{dir}"] = np.max(Es) * _e * 1e20

    
    if i == 3:
        # Only for (1, 1, 1)[1, 1, -2] stacking fault
        data["Disloc_SF_Form"] = float(Es[0, -1] - Es[0, 0]) * _e * 1e20 * 1000 # convert to mJ/m^2

add_info(active_model_name, data)