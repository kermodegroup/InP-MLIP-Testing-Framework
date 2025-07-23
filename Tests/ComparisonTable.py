import json
import os
import pandas
from active_model import plot_models
import numpy as np
import time
import datetime


index_dict = {
    "V0" : "V$_0$ (Å$^3$)",
    "a0" : "a$_0$ (Å)",
    "C11" : r"C$_{11}$ (GPa)",
    "C12" : r"C$_{12}$ (GPa)",
    "C44" : r"C$_{44}$ (GPa)",
    "ZB_In_Monoanti" : (r"P$\rightarrow$In", "Antisite"),
    "ZB_P_Monoanti" : (r"In$\rightarrow$P", "Antisite"),
    "ZB_Interstitial_In_Tetra" : ("In Tetrahedral", "Interstitial"),
    "ZB_Interstitial_In_Octa" : ("In Octahedral", "Interstitial"),
    "ZB_Interstitial_P_Dumbbell" : ("P Dumbbell", "Interstitial"),
    "ZB_In_Monovac" : r"In Vacancy",
    "ZB_P_Monovac" : r"P Vacancy",
    "E_(0, 0, 1)_(1, 0, 0)" : "(001)[100]", 
    "E_(0, 0, 1)_(1, 1, 0)" : "(001)[110]",
    "E_(1, 1, 0)_(1, -1, 0)" : "(110)[1-10]",
    "E_(1, 1, 1)_(1, 1, -2)" : "(111)[11-2]",
    "E_RMSE" : ("Binding Energy", "(meV/Atom)"),
    "F_RMSE" : r"Forces (meV/$\text{\AA}$)",
    "V_RMSE" : "Stresses (MPa)",
    "30 degree Partial Eform" : (r"$30^\circ$" + " Partial", "Formation Energy (eV)"),
    "90 degree Partial Eform" : (r"$90^\circ$" + " Partial", "Formation Energy (eV)"),
    "30 degree Partial dr" : (r"$30^\circ$" + " Partial RMSE on", "scaled positions ($10^{-2}$)"),
    "90 degree Partial dr" : (r"$90^\circ$" + " Partial RMSE on", "scaled positions ($10^{-2}$)"),
    "TT_Runtime" : ("Relative Inference Time", "(mm:ss; 1 CPU or 1 GPU)"),
    "Disloc_SF_Form" : ("(111)[11-2]", "ISF (mJ/m$^2$)") 
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
                "E_(1, 1, 1)_(1, 1, -2)",
                "Disloc_SF_Form"]

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

for key in index_dict.values():
    all_data[key] = {}

for model in models:
    mod_data = {}
    with open(f"../Test_Results/{model}.json", "r") as f:
        data = json.load(f)
    if model == "DFT":
        for key, val in index_dict.items():
            if key not in data.keys():
                all_data[val][r"RSCAN DFT"] = ["-", ""]
            else:
                all_data[val][r"RSCAN DFT"] = [np.round(data[key], 2), ""]
    else:
        for key, val in index_dict.items():
            if key not in data.keys():
                mod_data[val] = ["-", ""]
                continue
            if data[key] is None:
                all_data[val][model] = ["-", ""]
            elif key in dft_data.keys():
                err = data[key] - dft_data[key]
                # mod_data[key] = f"{data[key]:,.2f} ({'$+$' if err >=0 else '$-$'}{100 * np.abs(err) / np.abs(dft_data[key]):,.0f}\%)"
                all_data[val][model] = [np.round(data[key], 2), r"\textit{" + f"{'$+$' if err >=0 else '$-$'}{100 * np.abs(err) / np.abs(dft_data[key]):,.0f}\%" + r"}"]
            else:
                if key == "TT_Runtime":
                    d = data[key]
                    mins = int(np.floor(d/60))
                    secs = int(np.floor(d - 60 * mins))
                    rem = d - (secs + 60*mins)
                    all_data[val][model] = [f"{mins:02d}:{secs:02d}"]#{f'{rem:.3f}'[1:]}"
                else:
                    all_data[val][model] = [f"{data[key]:,.2f}", ""]
        all_data[model] = mod_data

# df = pandas.DataFrame.from_dict(all_data)
# df = df.rename(index=index_dict)

bulk_headers = [index_dict[key] for key in bulk_props]
pd_headers = [index_dict[key] for key in pd_props]
surf_headers = [index_dict[key] for key in surf_props]
rmse_headers = [index_dict[key] for key in rmse_props]
disloc_eform_headers = [index_dict[key] for key in disloc_props]


def print_table(caption, data, label, has_errors, end=False):
    print(
        r"""
\scriptsize
\begin{subtable}{\textwidth}
\caption{""" + caption + r"""}
\label{tbl:""" + label + r"""}
\begin{tabular}{@{}R{3.5cm}|""" + r"R{1cm}" * 7 + r"""@{}}
        """)

    keys = data.keys()

    #print(r"& RSCAN & & & & & & \\")
    print(" & DFT & " + " & ".join([model for model in list(data[list(keys)[0]].keys())[1:]]) + r"\\")
    print(r"\midrule \\")
    if has_errors:
        for key in keys:
            if type(key) == str:
                print(r"\multirow{2}{*}{" + str(key) + "} & " + " & ".join([str(val[0]) for val in data[key].values()]) + r"\\")
                print(r" & " + " & ".join([str(val[1]) for val in data[key].values()]) + r"\\")
            else:
                print(str(key[0]) + " & " + " & ".join([str(val[0]) for val in data[key].values()]) + r"\\")
                print(str(key[1]) + " & " + " & ".join([str(val[1]) for val in data[key].values()]) + r"\\")
            print(r"\vspace{-0.2cm} \\")
    else:
        for key in keys:
            if type(key) == tuple:
                print(str(key[0]) + " & " + " & ".join([r"\multirow{2}{*}{" + str(val[0]) + r"}" for val in data[key].values()]) + r" \\")
                print(str(key[1]) + " & & & & & & & " + r"\\")
            else:
                print(str(key) + r" & " + " & ".join([str(val[0]) for val in data[key].values()]) + r" \\")
            print(r"\vspace{-0.2cm} \\")

    print(r"""
\end{tabular}
\end{subtable}
\end{table*}""")
    if not end:
        print(r"""
\begin{table*}[htb]
\ContinuedFloat
    """)

print(r"""
\begin{table*}[htb]
\caption{Comparison of RSCAN DFT against the new InP ACE \& MACE models, as well as Vashishta \cite{Vashishta} and SNAP \cite{SNAP} InP models  and MP0 and MPA foundation models \cite{MP0} from existing literature}
\label{tab:comp}""")

print_table("Bulk Properties", {key : all_data[key] for key in bulk_headers}, "Bulk", True)
print_table("Point Defect Formation Energies (eV; $\mu_{In} = \mu_{P}$)", {key : all_data[key] for key in pd_headers}, "PointDefect", True)
print_table("Stacking Fault Energies (J/m$^2$)", {key : all_data[key] for key in surf_headers}, "StackFault", True)
print_table("Dislocation Quadrupole Properties", {key : all_data[key] for key in disloc_eform_headers}, "Disloc", True)
print_table("Dataset Performance", {key : all_data[key] for key in rmse_headers}, "Dataset", False, True)
