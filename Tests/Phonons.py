from ase.io import read
from ase.phonons import Phonons
from active_model import *

calc_name = active_model_name
calc = get_model(active_model_name)

delta = 0.1
mul = 8

bulk = read("Misc_Reference/Phonon_bulk.xyz")


path = bulk.cell.bandpath('GXGL', npoints=100)

name = f"../Test_Results/{calc_name}/Phonons"

bulk.calc = calc
ph = Phonons(bulk, calc, supercell=[mul]*3, delta=delta, name=name)

ph.run()