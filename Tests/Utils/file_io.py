from ase.io import read as ase_read, write
import os

fpath = "/home/eng/phrbqc/GitHub/InPDislocs/Data/Dataset"


def read(path, index=":"):
    return ase_read(os.path.normpath(fpath + os.sep + path), index=index)