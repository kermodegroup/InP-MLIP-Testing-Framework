import os
from Utils.models import *
from Utils.utils import get_bulk

## Change the value of active_model_name to change the model used in each of the tests

active_model_name = "ACE"
active_model_name = "MACE"
active_model_name = "SNAP"
active_model_name = "Vashishta"
active_model_name = "MP0"
active_model_name = "MPA"

## List of models (in a consistent order) to plot in each of the plotting functions
plot_models = ["DFT", "Vashishta", "SNAP", "MP0", "MPA", "ACE", "MACE"]

models = {
    "ACE" : ACE,
    "MACE" : MACE,
    "SNAP" : SNAP,
    "Vashishta" : Vashishta,
    "MP0" : MP0,
    "MPA" : MPA
}

def get_model(model_name):
    os.makedirs("../Test_Plots" + os.sep + model_name, exist_ok=True)
    os.makedirs("../Test_Results" + os.sep + model_name, exist_ok=True)    
    return models[model_name]()