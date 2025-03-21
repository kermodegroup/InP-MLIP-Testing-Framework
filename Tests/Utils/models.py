import os
from ase.calculators.lammpslib import LAMMPSlib
import torch
from ase.calculators.calculator import BaseCalculator


class LMPMIX(BaseCalculator):
    '''
    Wrapper calculator for dealing with LAMMPS calculators aplied to structures with missing elements
    (e.g. InP potential applied to isolated P)
    '''
    def __init__(self, base_func, *funcargs, **funckwargs):
        self.calcs = {
            "In" : base_func(*funcargs, In=True, P=False, **funckwargs),
            "P" : base_func(*funcargs, In=False, P=True, **funckwargs),
            "InP" : base_func(*funcargs, In=True, P=True, **funckwargs),
        }
        super().__init__()

        self.implemented_properties = self.calcs["InP"].implemented_properties

    def calculate(self, atoms, properties, system_changes):
        spec = atoms.get_chemical_symbols()

        In = "In" in spec
        P = "P" in spec

        if In and not P:
            calc = self.calcs["In"]
        elif not In and P:
            calc = self.calcs["P"]
        else:
            calc = self.calcs["InP"]

        calc.calculate(atoms, properties, system_changes)
        self.results = calc.results

fpath = "/home/eng/phrbqc/GitHub/InPDislocs/Potentials/"

def ACE(lammps=True):
    path = fpath + f"ACE"
    
    if lammps:
        lammps_commands = [
            f"pair_style      hybrid/overlay pace table spline 6000",
            f"pair_coeff      * * pace {path}/InP_ACE.yace P In",
            f"pair_coeff      1 1 table {path}/InP_ACE_pairpot.table P_P",
            f"pair_coeff      1 2 table {path}/InP_ACE_pairpot.table P_In",
            f"pair_coeff      2 2 table {path}/InP_ACE_pairpot.table In_In"]
        return LAMMPSlib(lmpcmds=lammps_commands, keep_alive=True, atom_types={"P": 1, "In" : 2})
    else:
        import pyjulip
        calc = pyjulip.ACE1(path + os.sep + "InP_ACE.json")
        return calc

def MACE():
    from mace.calculators import MACECalculator

    if torch.cuda.is_available():
        print(f"MACE using GPU")
        device = "cuda"
    else:
        print(f"MACE using CPU")
        device = "cpu"

    pot_path = fpath + os.sep + "MACE/InP_MACE_stagetwo.model"

    calculator = MACECalculator(model_paths=pot_path, device=device)
    return calculator


def MP0():
    from mace.calculators import mace_mp

    if torch.cuda.is_available():
        print(f"MP0 using GPU")
        device = "cuda"
    else:
        print(f"MP0 using CPU")
        device = "cpu"

    return mace_mp(model="medium", default_dtype="float64", device=device)

def MPA():
    from mace.calculators import mace_mp
    #from mace_jax import MACE

    if torch.cuda.is_available():
        print(f"MPA using GPU")
        device = "cuda"
    else:
        print(f"MPA using CPU")
        device = "cpu"

    return mace_mp(model="medium-mpa-0", default_dtype="float64", device=device)

def Vashishta_Base(In=True, P=True):
    elem = ""
    if In:
        elem += "In "
    if P:
        elem += "P "

    lmpcmds = ["pair_style vashishta",
               f"pair_coeff * * {fpath}/Vashishta/InP.vashishta {elem}"]
    lmp = LAMMPSlib(lmpcmds=lmpcmds, log_file="run.log", keep_alive=True, atom_types={
                    "In": 1, "P": 2})
    return lmp

# def Vashishta():
#     lmpcmds = ["pair_style vashishta",
#                f"pair_coeff * * {fpath}/Vashishta/InP.vashishta In P"]
#     lmp = LAMMPSlib(lmpcmds=lmpcmds, log_file="run.log", keep_alive=True, atom_types={
#                     "In": 1, "P": 2})
#     return lmp


def Vashishta():
    return LMPMIX(Vashishta_Base)

def MLIAP_Base(In=True, P=True):
    elem = ""
    if In:
        elem += "In "
    if P:
        elem += "P "

    lmpcmds = [
        f"pair_style mliap model linear {fpath}/MLIAP/InP_JCPA2020.mliap.model descriptor sna {fpath}/MLIAP/InP_JCPA2020.mliap.descriptor", f"pair_coeff * * {elem}"]
    lmp = LAMMPSlib(lmpcmds=lmpcmds, log_file="run.log", keep_alive=True, atom_types={"In": 1, "P": 2}
    )
    return lmp

def MLIAP():
    return LMPMIX(MLIAP_Base)

def SNAP_Base(In=True, P=True):
    if P == False:
        lmpcmds = ["pair_style hybrid/overlay zbl 4 4.2 snap",
                   f"pair_coeff 1 1 zbl 49 49",
                   f"pair_coeff * * snap {fpath}/SNAP/InP_pot.snapcoeff {fpath}/SNAP/InP_pot.snapparam In P"]
    elif In == False:
        lmpcmds = ["pair_style hybrid/overlay zbl 4 4.2 snap",
                   f"pair_coeff 2 2 zbl 15 15",
                   f"pair_coeff * * snap {fpath}/SNAP/InP_pot.snapcoeff {fpath}/SNAP/InP_pot.snapparam In P"]
    else:
        lmpcmds = ["pair_style hybrid/overlay zbl 4 4.2 snap",
                   f"pair_coeff 1 1 zbl 49 49",
                   f"pair_coeff 1 2 zbl 49 15",
                   f"pair_coeff 2 2 zbl 15 15",
                   f"pair_coeff * * snap {fpath}/SNAP/InP_pot.snapcoeff {fpath}/SNAP/InP_pot.snapparam In P"]

    lmp = LAMMPSlib(lmpcmds=lmpcmds, atom_types={
                    "In": 1, "P": 2}, log_file="run.log", keep_alive=True)
    return lmp

def SNAP():
    return LMPMIX(SNAP_Base)