import json
import os
import pandas
from active_model import plot_models
import numpy as np
import time
import datetime

nepochs = {
    "MACE7" : "300 (299)",
    "MACEFRZ20" : "700 (488)",
    "MACEFRZ100" : "700 (628)",
    "MACEFRZ20A" : "700 (624)",
    "MACEFRZ100A" : "70 (67)"
}

train_times = {
    "ACE13" : "08:59:07",
    "MACE7" : "4:11:37",
    "MACEFRZ20" : "00:58:16",
    "MACEFRZ100" : "03:59:32",
    "MACEFRZ20A" : "02:21:27",
    "MACEFRZ100A" : "1:01:34"
}

index_dict = {
    "V0" : "V0 (Å$^3$)",
    "a0" : "a0 (Å)",
    "C11" : "C11 (GPa)",
    "C12" : "C12 (GPa)",
    "C44" : "C44 (GPa)",
    "ZB_In_Monoanti" : r"P$\rightarrow$In Antisite",
    "ZB_P_Monoanti" : r"In$\rightarrow$P Antisite",
    "ZB_Interstitial_In_Tetra" : "In Tetrahedral Interstitial",
    "ZB_Interstitial_In_Octa" : "In Octahedral Interstitial",
    "ZB_Interstitial_P_Dumbbell" : "P Dumbbell Interstitial",
    "ZB_In_Monovac" : r"In Vacancy",
    "ZB_P_Monovac" : r"P Vacancy",
    "E_(0, 0, 1)_(1, 0, 0)" : "(001)[100]", 
    "E_(0, 0, 1)_(1, 1, 0)" : "(001)[110]",
    "E_(1, 1, 0)_(1, -1, 0)" : "(110)[1-10]",
    "E_(1, 1, 1)_(1, 1, -2)" : "(111)[11-2]",
    "E_RMSE" : "Binding Energy (meV/Atom)",
    "F_RMSE" : "Forces (meV/Ang)",
    "V_RMSE" : "Stresses (MPa)",
    "30 degree Partial Eform" : r"$30^\circ$" + " Partial Formation Energy (eV)",
    "90 degree Partial Eform" : r"$90^\circ$" + " Partial Formation Energy (eV)",
    "30 degree Partial dr" : r"$30^\circ$" + " Partial RMSE on scaled positions ($10^{-2}$)",
    "90 degree Partial dr" : r"$90^\circ$" + " Partial RMSE on scaled positions ($10^{-2}$)",
    "TT_Runtime" : "Time for single Dataset Pass (mm:ss; 1 CPU or 1 GPU)",
    "TrainTime" : "Model Training Time",
    "Nepochs" : "Number of training epochs"
}

bulk_props = ["V0", "a0", "C11", "C12", "C44"]
pd_props = ["ZB_In_Monoanti",
    "ZB_P_Monoanti",
    "ZB_Interstitial_In_Tetra",
    "ZB_Interstitial_In_Octa",
    "ZB_Interstitial_P_Dumbbell",
    "ZB_In_Monovac",
    "ZB_P_Monovac"]

surf_props = ["E_(0, 0, 1)_(1, 0, 0)", 
                "E_(0, 0, 1)_(1, 1, 0)",
                "E_(1, 1, 0)_(1, -1, 0)",
                "E_(1, 1, 1)_(1, 1, -2)"]

rmse_props = [
    "E_RMSE", "F_RMSE", "V_RMSE", "TT_Runtime"
]

disloc_props = [
    "30 degree Partial Eform",
    "30 degree Partial dr",
    "90 degree Partial Eform",
    "90 degree Partial dr",
]

models = plot_models


if "MACEFRZ100A" in models:
    rmse_props.append("Nepochs")
    rmse_props.append("TrainTime")

all_data = {}

new_keys = {}

for key, tkey in index_dict.items():
    if type(tkey) == tuple:
        new_key = tkey
    else:
        new_key = ("", tkey)
    new_keys[key] = new_key

with open("../Test_Results/DFT.json", "r") as f:
    dft_data = json.load(f)

for model in models:
    mod_data = {}
    with open(f"../Test_Results/{model}.json", "r") as f:
        data = json.load(f)
    if model == "DFT":
        for key in index_dict.keys():
            if key not in data.keys():
                mod_data[key] = "-"
            else:
                mod_data[key] = f"{data[key]:,.2f}"
        all_data[r"RSCAN DFT"] = mod_data
    else:
        for key in index_dict.keys():
            if key not in data.keys():
                mod_data[key] = "-"
                continue
            if data[key] is None:
                mod_data[key] = "-"
            elif key in dft_data.keys():
                err = data[key] - dft_data[key]
                mod_data[key] = f"{data[key]:,.2f} ({'$+$' if err >=0 else '$-$'}{100 * np.abs(err) / np.abs(dft_data[key]):,.0f}\%)"
            else:
                if key == "TT_Runtime":
                    d = data[key]
                    mins = int(np.floor(d/60))
                    secs = int(np.floor(d - 60 * mins))
                    rem = d - (secs + 60*mins)
                    mod_data[key] = f"{mins:02d}:{secs:02d}"#{f'{rem:.3f}'[1:]}"
                else:
                    mod_data[key] = f"{data[key]:,.2f}"

        if model in nepochs.keys():
            mod_data["Nepochs"] = nepochs[model]

        if model in train_times.keys():
            mod_data["TrainTime"] = train_times[model]

        all_data[model] = mod_data

df = pandas.DataFrame.from_dict(all_data)
df = df.rename(index=index_dict)

bulk_headers = [index_dict[key] for key in bulk_props]
pd_headers = [index_dict[key] for key in pd_props]
surf_headers = [index_dict[key] for key in surf_props]
rmse_headers = [index_dict[key] for key in rmse_props]
disloc_eform_headers = [index_dict[key] for key in disloc_props]

def modi_table(table):
    return r"\begin{tabular*}{\textwidth}{" + r"p{0.168\textwidth}|" + "".join([r"p{0.097\textwidth}" for i in range(7)]) + "}" + table[25:-14] + r"\end{tabular*}"

print(r"\scriptsize")
print(r"\begin{subtable}{\textwidth}")
print(r"\caption{Bulk Properties}")
print(modi_table(df.loc[bulk_headers].to_latex(header=True, index=True)))
print(r"\end{subtable}")
print(r"\end{table*}")
print(r"\begin{table*}")
print(r"\ContinuedFloat")
print(r"\scriptsize")

print(r"\begin{subtable}{\textwidth}")
print(r"\caption{Point Defect Formation Energies (eV; $\mu_{In} = \mu_{P}$)}")
print(modi_table(df.loc[pd_headers].to_latex(header=True, index=True)))
print(r"\end{subtable}")
print(r"\end{table*}")
print(r"\begin{table*}")
print(r"\ContinuedFloat")
print(r"\scriptsize")


print(r"\begin{subtable}{\textwidth}")
print(r"\caption{Stacking Fault Energies (J/m$^2$)}")
print(modi_table(df.loc[surf_headers].to_latex(header=True, index=True)))
print(r"\end{subtable}")
print(r"\end{table*}")
print(r"\begin{table*}")
print(r"\ContinuedFloat")
print(r"\scriptsize")

print(r"\begin{subtable}{\textwidth}")
print(r"\caption{Dislocation Quadrupole Formation Energies (eV)}")
print(modi_table(df.loc[disloc_eform_headers].to_latex(header=True, index=True)))
print(r"\end{subtable}")
#print(df.to_latex())
print(r"\end{table*}")
print(r"\begin{table*}")
print(r"\ContinuedFloat")
print(r"\scriptsize")


print(r"\begin{subtable}{\textwidth}")
print(r"\caption{Dataset Performance}")
print(modi_table(df.loc[rmse_headers].to_latex(header=True, index=True)))
print(r"\end{subtable}")