from matscipy.dislocation import Quadrupole, DiamondGlide30degreePartial, DiamondGlide90degreePartial, DiamondGlide60Degree, DiamondGlideScrew
from utils import get_bulk
from models import *
from quippy.nye_tensor import nye_tensor
from ase.optimize.precon import PreconLBFGS
from ase.io import read, write
import numpy as np
from jsondata import add_info
from active_model import *
from matscipy.neighbours import mic
from neb_core import do_NEB

calc_name = active_model_name
calc = get_model(active_model_name)

disloc_names = ["30 degree Partial", "90 degree Partial"]

short_names = ["30deg", "90deg"]

dft_structs = [read("DFT_Quadrupoles/DFT_Quads_0/DFT_Quads_0.geom", index="-1"), read("DFT_Quadrupoles/DFT_Quads_1/DFT_Quads_1.geom", index="-1")]

structs = []
calc_data = {}

dft_a0 = 5.901273599999999
nims=15

for k, disloc in enumerate([DiamondGlide30degreePartial, DiamondGlide90degreePartial]):
    a = get_bulk(calc)
    ref_bulk = a[0]
    d = Quadrupole(disloc, *a)

    fname = f"../Saved_Data/{calc_name}/{short_names[k]}_Quadrupole_Migration_structs.xyz"

    if os.path.exists(fname):
        ims = read(fname, index=":")
    else:
        ims = d.build_glide_quadrupoles(nims, glide_left=False, glide_right=True, glide_separation=4, self_consistent=False)
        #ims = read(f"../Saved_Data/ACE13/{short_names[k]}_Quadrupole_Migration_structs.xyz", index=":")
    neb, e = do_NEB(calc, ims, ims[-1], nims=nims, ci=False, ftol=1e-3, neb_ftol=7e-2, max_nsteps=500, mic=True, idpp=False, refine=False)

    structs.extend(ims) 

    write(f"../Saved_Data/{calc_name}/{short_names[k]}_Quadrupole_Migration_structs.xyz", neb.images)

add_info(calc_name, calc_data)