from ase.dft.kpoints import BandPath
from scipy.constants import h, e
import numpy as np
from ase.constraints import UnitCellFilter
from ase.io import read as ase_read, write
from file_io import read
from ase.phonons import Phonons
from models import *
from active_model import *

calc_name = active_model_name
calc = get_model(active_model_name)

delta = 0.1
mul = 8

bulk = ase_read("Phonon_bulk.xyz")


path = bulk.cell.bandpath('GXGL', npoints=100)

name = f"../Saved_Data/{calc_name}/Phonons"

bulk.calc = calc
ph = Phonons(bulk, calc, supercell=[mul]*3, delta=delta, name=name)

ph.run()