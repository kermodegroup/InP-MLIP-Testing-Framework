from file_io import read
from castep import castep, get_even_kpts
import numpy as np

bulk = read("Bulk/ZB_Bulk.xyz", index="-1")
bulk.rattle(1e-1)

a = bulk.cell[0, 0]

encuts = [600, 750, 900, 1050, 1200]
N_k = np.array([5, 6, 7, 8, 9])

offsets = 0.5 / N_k

for encut in encuts:
    for i, N in enumerate(N_k):
        name = f"{encut}_{N}"
        calc = castep(bulk, directory=name, relax=False, offset=True, e_cut=encut)
        calc.param.calculate_stress = False

        calc.cell.kpoint_mp_grid = np.array([N] * 3)
        calc.cell.kpoint_mp_offset = np.array([offsets[i]] * 3)

        print(encut, N, offsets[i])
        bulk.calc = calc
        try:
            bulk.get_potential_energy()
        except AttributeError:
            pass

### Accurate Reference
encut = 1500
N = 12
offset = 0.5 / N

name = f"{encut}_{N}"
calc = castep(bulk, directory=name, relax=False, offset=True, e_cut=encut)
calc.param.calculate_stress = False

calc.cell.kpoint_mp_grid = np.array([N] * 3)
calc.cell.kpoint_mp_offset = np.array([offset] * 3)

print(encut, N, offset)

bulk.calc = calc
bulk.get_potential_energy()