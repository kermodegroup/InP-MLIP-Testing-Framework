import os
from models import *
from utils import get_bulk


gap = 0
ace = 1
mace = 2

model = mace

gap_num = 16
ace_num =  13
mace_num = 7

active_gap = "GAP" + str(gap_num)
active_ace = "ACE" + str(ace_num)
active_mace = "MACE" + str(mace_num)

plot_models = ["DFT", f"ACE{ace_num}", f"MACE{mace_num}", "MACEFRZ20", "MACEFRZ100", "MACEFRZ20A", "MACEFRZ100A"]
plot_models = ["DFT", "Vashishta", "SNAP", "MP0", "MPA", f"ACE{ace_num}", f"MACE{mace_num}"]


active_model_name = [active_gap, active_ace, active_mace][model]


#active_model_name = "Vashishta"
#active_model_name = "SNAP"
#active_model_name = "MP0"
#active_model_name = "MPA"

#active_model_name = "MACEFRZ20"
#active_model_name = "MACEFRZ100"

#active_model_name = "MACEFRZ20A"
#active_model_name = "MACEFRZ100A"

def get_model(model_name=active_model_name):

    if "GAP" in model_name:
        num = model_name[3:]
        active_model = GAP(num)
    elif model_name[:3] == "ACE":
        num = model_name[3:]
        active_model = ACE(num)
    elif "SNAP" in model_name:
        active_model = SNAP()
    elif "Vashishta" in model_name:
        active_model = Vashishta()
    elif "MP0" in model_name:
        active_model = MP0()
    elif "MPA" in model_name:
        active_model = MPA()
    else:
        num = model_name[4:]
        active_model = MACE(num)

    os.makedirs("../Plots" + os.sep + model_name, exist_ok=True)

    os.makedirs("../Plots/" + os.sep + model_name +
                os.sep + "TT_Config_Types", exist_ok=True)
    return active_model
