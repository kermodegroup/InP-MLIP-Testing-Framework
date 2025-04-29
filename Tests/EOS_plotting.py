import os
from active_model import *
import matplotlib.pyplot as plt
import numpy as np
import json
from ase.io import read

calcs = [model for model in plot_models if model != "DFT"]

markers = [".", "x", "o", "s", "*", "+", "D", "^", "<", ">", "v", "H", "d"]

## DFT
with open("../Test_Results/DFT/EOS.json", "r") as f:
    d = json.load(f)

fig, ax = plt.subplots(2, figsize=(10, 8), sharex=True)

ax[0].scatter(d["V_ZB"], d["E_ZB"], label="RSCAN DFT", color="k", marker=".")
ax[1].scatter(d["V_WZ"], d["E_WZ"], label="RSCAN DFT", color="k", marker=".")


# LDA DFT
zb_dir = "DFT_Reference/JPCA_EOS/EOS"
wz_dir = "DFT_Reference/JPCA_EOS/EOS_WZ"

zb_dft = []
for file in os.listdir(zb_dir):
    zb_dft.extend(read(zb_dir + os.sep + file, index=":"))


wz_dft = []
for file in os.listdir(wz_dir):
    wz_dft.extend(read(wz_dir + os.sep + file, index=":"))

v_zb = [ats.get_volume()/len(ats) for ats in zb_dft]
v_wz = [ats.get_volume()/len(ats) for ats in wz_dft]

Es_zb = np.array([ats.get_potential_energy()/len(ats) for ats in zb_dft])
Es_wz = np.array([ats.get_potential_energy()/len(ats) for ats in wz_dft])

E0 = np.min(Es_zb)
Es_zb -= E0
Es_wz -= E0

ax[0].scatter(v_zb, Es_zb, label="LDA DFT", color="k", marker="x")
ax[1].scatter(v_wz, Es_wz, label="LDA DFT", color="k", marker="x")

for i, calc in enumerate(calcs):
    if not os.path.exists(f"../Test_Results/{calc}/EOS.json"):
        continue
    with open(f"../Test_Results/{calc}/EOS.json", "r") as f:
        d = json.load(f)
    ax[0].plot(d["V_ZB"], d["E_ZB"], label=f"{calc}", color=f"C{i}")
    ax[1].plot(d["V_WZ"], d["E_WZ"], label=f"{calc}", color=f"C{i}")

ax[1].set_xlabel("Volume Per Atom ($Å^3$)")
ax[0].set_ylabel("Energy Per Atom (eV)")
ax[1].set_ylabel("Energy Per Atom (eV)")

ax[0].set_xlim(13, 30)
ax[0].set_ylim(-0.1, 6.3)

ax[1].set_xlim(13, 30)
ax[1].set_ylim(-0.1, 6.3)

plt.legend()

plt.tight_layout()
plt.savefig("../Test_Plots/EOS.png", dpi=200)

plt.clf()

## DFT
zoomed_vol_lims = [23.5, 27]
ylims = [-1, 44]

with open("../Test_Results/DFT/EOS.json", "r") as f:
    d = json.load(f)

fig, ax = plt.subplots(2, figsize=(8, 10), sharex=True)

ax[0].scatter(d["V_ZB"], np.array(d["E_ZB"]) * 1000, label="RSCAN DFT", color="k", marker="x", zorder=50, s=50)
ax[1].scatter(d["V_WZ"], np.array(d["E_WZ"]) * 1000, label="RSCAN DFT", color="k", marker="x", zorder=50, s=50)

m = np.argmin(d["E_ZB"])
ax[0].axvline(d["V_ZB"][m], color="k", alpha=0.6)
m = np.argmin(d["E_WZ"])
ax[1].axvline(d["V_WZ"][m], color="k", alpha=0.6)


# LDA DFT
ax[0].scatter(v_zb, Es_zb*1000, label="LDA DFT", color="k", marker=".", s=40)
ax[1].scatter(v_wz, Es_wz*1000, label="LDA DFT", color="k", marker=".", s=40)


for i, calc in enumerate(calcs):
    if not os.path.exists(f"../Test_Results/{calc}/EOS.json"):
        continue
    with open(f"../Test_Results/{calc}/EOS.json", "r") as f:
        d = json.load(f)
    ax[0].plot(d["V_ZB"], np.array(d["E_ZB"]) * 1000, label=f"{calc}", color=f"C{i}")
    ax[1].plot(d["V_WZ"], np.array(d["E_WZ"]) * 1000, label=f"{calc}", color=f"C{i}")

    m = np.argmin(d["E_ZB"])
    ax[0].axvline(d["V_ZB"][m], color=f"C{i}", alpha=0.6)
    m = np.argmin(d["E_WZ"])
    ax[1].axvline(d["V_WZ"][m], color=f"C{i}", alpha=0.6)

ax[0].set_xlim(*zoomed_vol_lims)
ax[0].set_ylim(*ylims)
ax[1].set_xlim(*zoomed_vol_lims)
ax[1].set_ylim(*ylims)

ax[1].set_xlabel("Volume Per Atom ($Å^3$)")
ax[0].set_ylabel("Relative Energy Per Atom (meV)")
ax[1].set_ylabel("Relative Energy Per Atom (meV)")

ax[0].set_title("Zincblende")
ax[1].set_title("Wurtzite")
fig.suptitle("Equation of State comparison of several potentials")

ax[0].legend()
plt.tight_layout()
plt.savefig("../Test_Plots/EOS_Zoomed.png", dpi=200)

