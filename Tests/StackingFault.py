from active_model import get_model, active_model_name
from ase.io import write
from matscipy.gamma_surface import StackingFault
import os
from ase.units import _e
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from Utils.jsondata import add_info
from Utils.utils import get_bulk

nims = 81

planes = [(0, 0, 1), (0, 0, 1), (1, 1, 0), (1, 1, 1)]
dirs = [(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 1, -2)]
zreps = [3, 6, 6, 2]
max_lims = [1, 1, 1, 1]

#active_model_name = "GAP16"

active_model = get_model(active_model_name)

bulk = get_bulk(active_model, elast=False)

print("start")

data = {}

print(active_model_name)

for i in range(len(planes)):
    plane = planes[i]
    dir = dirs[i]
    zr = zreps[i]
    yl = max_lims[i]

    fault = StackingFault(bulk, plane, dir)
    fault.generate_images(nims, z_reps=zr, path_ylims=[0, yl])

    print(plane, dir, len(fault.images[0]))
    os.makedirs(f"../Test_Results/{active_model_name}/", exist_ok=True)
    if os.path.exists(f"../Test_Results/{active_model_name}/StackingFaultStructs_{plane}_{dir}.xyz"):
        continue
        #ims = ase_read(f"../Test_Results/{active_model_name}/StackingFaultStructs_{plane}_{dir}.xyz", index=":")
        #if len(ims) == nims and len(ims[0]) == len(fault.images[0]):
        #    fault.images = ims

    Es = fault.get_energy_densities(active_model, relax=True, cell_relax=False)

    for image in fault.images:
        image.calc = active_model
        E = image.get_potential_energy()
        image.calc = SinglePointCalculator(image, energy=E)

    write(f"../Test_Results/{active_model_name}/StackingFaultStructs_{plane}_{dir}.xyz", fault.images)

    data[f"E_{plane}_{dir}"] = np.max(Es) * _e * 1e20

add_info(active_model_name, data)