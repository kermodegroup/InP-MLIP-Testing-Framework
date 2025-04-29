import os
from ase.calculators.lammpslib import LAMMPSlib
import torch
from ase.calculators.calculator import BaseCalculator

import pathlib
file_root = os.path.dirname(os.path.abspath(__file__))

fpath = os.path.normpath(file_root + "../../../Potentials/")

def ACE(lammps=True):
    path = fpath + os.sep + f"ACE"
    
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

    path = fpath + os.sep + "MACE/InP_MACE_stagetwo.model"

    if not os.path.exists(path):
        # Fallback for no stagetwo (e.g. for multihead models)
        path = fpath + os.sep + "MACE/InP_MACE.model"

    calculator = MACECalculator(model_paths=path, device=device)
    return calculator


def __MACE_MP(mp_name):
    from mace.calculators import mace_mp

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return mace_mp(model=mp_name, default_dtype="float64", device=device)
def MP0():
    return __MACE_MP("medium")

def MPA():
    return __MACE_MP("medium-mpa-0")

def Vashishta():
    lmpcmds = ["pair_style vashishta",
               f"pair_coeff * * {fpath}/Vashishta/InP.vashishta In P"]
    lmp = LAMMPSlib(lmpcmds=lmpcmds, log_file="run.log", keep_alive=True, atom_types={
                    "In": 1, "P": 2})
    return lmp

def SNAP():
    lmpcmds = lmpcmds = ["pair_style hybrid/overlay zbl 4 4.2 snap",
                   f"pair_coeff 1 1 zbl 49 49",
                   f"pair_coeff 1 2 zbl 49 15",
                   f"pair_coeff 2 2 zbl 15 15",
                   f"pair_coeff * * snap {fpath}/SNAP/InP_pot.snapcoeff {fpath}/SNAP/InP_pot.snapparam In P"]
    lmp = LAMMPSlib(lmpcmds=lmpcmds, log_file="run.log", keep_alive=True, atom_types={
                    "In": 1, "P": 2})
    return lmp
