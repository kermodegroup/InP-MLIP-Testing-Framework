from active_model import *
from Utils.file_io import read
from tqdm import tqdm
import numpy as np
import json
import os
from ase.units import GPa
from Utils.jsondata import add_info
from time import time

data_dir = "../Test_Results"

def _setup():

    dataset = read("../InP_Dataset.xyz", index=":")


    iso_In = read("IsolatedAtom/Iso_In.xyz", index="-1")
    iso_P = read("IsolatedAtom/Iso_P.xyz", index="-1")
    config_types_rmses = {}

    conf_raw_data = {}

    dataset_rmses = {}

    dft_E_In = iso_In.get_potential_energy()
    dft_E_P = iso_P.get_potential_energy()

    conf_raw_data["energy"] = {}
    conf_raw_data["forces"] = {}
    conf_raw_data["stresses"] = {}
    conf_raw_data["names"] = {}
    conf_raw_data["Sigmas"] = {}
    conf_raw_data["counts"] = {}

    for image in dataset:
        config_type = image.info["config_type"]

        conf_raw_data["energy"][config_type] = []
        conf_raw_data["forces"][config_type] = []
        conf_raw_data["stresses"][config_type] = []
        conf_raw_data["names"][config_type] = []
        conf_raw_data["Sigmas"][config_type] = [0] * 3
        conf_raw_data["counts"][config_type] = [0] * 3

        config_types_rmses[config_type] = {}
    
    return [dataset, iso_In, iso_P, dft_E_In, dft_E_P, config_types_rmses, conf_raw_data, dataset_rmses]


def test_calc(calc, x):
    dataset, iso_In, iso_P, dft_E_In, dft_E_P, config_types_rmses, conf_raw_data, dataset_rmses = x

    iso_In.calc = calc
    calc_E_In = iso_In.get_potential_energy()
    iso_P.calc = calc
    calc_E_P = iso_P.get_potential_energy()

    s_err = np.zeros(1)
    f_err = np.zeros(1)

    for image in tqdm(dataset):
        if len(image) == 1:
            continue # skip iso ats
        has_energy = True
        has_force = True
        has_stress = True

        config_type = image.info["config_type"]

        spec = np.array(image.get_chemical_symbols())
        n_In = sum(spec == "In")
        n_P = sum(spec == "P")
        dft_E0 = n_In * dft_E_In + n_P * dft_E_P
        calc_E0 = n_In * calc_E_In + n_P * calc_E_P

        try:
            e_true = image.get_potential_energy() - dft_E0
        except:
            has_energy = False
        
        try:
            f_true = image.get_forces()
        except:
            has_force = False

        try:
            s_true = image.get_stress()
        except:
            has_stress = False

        image.calc = calc

        if has_energy:
            e_err = (e_true - (image.get_potential_energy() - calc_E0)) / len(image)

            conf_raw_data["energy"][config_type].append(e_err * 1000)
            conf_raw_data["Sigmas"][config_type][0] += e_err * \
                ((e_true)/len(image))

        if has_force:
            f_err = (f_true - image.get_forces()).flatten()

            conf_raw_data["forces"][config_type].extend(f_err * 1000)
            conf_raw_data["Sigmas"][config_type][1] += np.sum(
                f_err * f_true.flatten())

        if has_stress:
            s_err = (s_true - image.get_stress()).flatten()
            conf_raw_data["stresses"][config_type].extend(s_err / GPa * 1000)
            conf_raw_data["Sigmas"][config_type][2] += np.sum(
                s_err * s_true.flatten())

        local_counts = np.array(
            [has_energy, has_force * f_err.shape[0], has_stress * s_err.shape[0]], dtype=int)

        for i in range(3):
            conf_raw_data["counts"][config_type][i] += int(local_counts[i])

    # E, F, V RMSE Data
    config_types_rmses["Total"] = {}
    for var_key in ["energy", "forces", "stresses"]:
        accumulation = []
        conf_keys = conf_raw_data[var_key].keys()

        for config_type in conf_keys:
            all_data = np.array(conf_raw_data[var_key][config_type])

            rmse = np.sqrt(np.average(all_data ** 2))

            config_types_rmses[config_type][var_key] = rmse

            accumulation.extend(all_data)

        config_types_rmses["Total"][var_key] = np.sqrt(
            np.average(np.array(accumulation) ** 2))

    # E, F, V Counts
    accumulation = np.zeros(3)
    conf_keys = conf_raw_data[var_key].keys()

    for config_type in conf_keys:
        counts = conf_raw_data["counts"][config_type]
        accumulation += counts

        config_types_rmses[config_type]["Energy_counts"] = int(counts[0])
        config_types_rmses[config_type]["Force_counts"] = int(counts[1])
        config_types_rmses[config_type]["Virial_counts"] = int(counts[2])

    config_types_rmses["Total"]["Energy_counts"] = int(accumulation[0])
    config_types_rmses["Total"]["Force_counts"] = int(accumulation[1])
    config_types_rmses["Total"]["Virial_counts"] = int(accumulation[2])

    tot_counts = accumulation.copy()

    # E, F, V Sigmas
    accumulation = np.zeros(3)
    conf_keys = conf_raw_data[var_key].keys()

    for config_type in conf_keys:
        sigmas = conf_raw_data["Sigmas"][config_type]
        counts = conf_raw_data["counts"][config_type]
        accumulation += sigmas

        try:
            config_types_rmses[config_type]["Energy_sigma"] = float(
                np.sqrt(np.abs(sigmas[0] / counts[0]))) * 1000
        except:
            config_types_rmses[config_type]["Energy_sigma"] = np.inf

        try:
            config_types_rmses[config_type]["Force_sigma"] = float(
                np.sqrt(np.abs(sigmas[1] / counts[1]))) * 1000
        except:
            config_types_rmses[config_type]["Force_sigma"] = np.inf

        if counts[2]:
            config_types_rmses[config_type]["Virial_sigma"] = float(
                np.sqrt(np.abs(sigmas[2] / counts[2]))) / GPa * 1000

    config_types_rmses["Total"]["Energy_sigma"] = float(
        np.sqrt(np.abs(accumulation[0] / tot_counts[0]))) * 1000
    config_types_rmses["Total"]["Force_sigma"] = float(
        np.sqrt(np.abs(accumulation[1] / tot_counts[1]))) * 1000

    if tot_counts[2]:
        config_types_rmses["Total"]["Virial_sigma"] = float(
            np.sqrt(np.abs(accumulation[2] / tot_counts[2]))) / GPa * 1000

    return config_types_rmses, conf_raw_data


def save(calc_name, config_types_rmses, conf_raw_data):
    with open(data_dir + os.sep + "TT_RMSE" + os.sep + calc_name + "_TT_RMSE.json", "w") as f:
        json.dump(config_types_rmses, f, indent=4)

    with open(data_dir + os.sep + "TT_RMSE" + os.sep + calc_name + "_RAW_TT_DIFFS.json", "w") as f:
        json.dump(conf_raw_data, f, indent=4)


x = _setup()

config_types_rmses, conf_raw_data = test_calc(get_model(active_model_name), x)

save(active_model_name, config_types_rmses, conf_raw_data)


data = {
    "E_RMSE" : config_types_rmses["Total"]["energy"],
    "F_RMSE" : config_types_rmses["Total"]["forces"],
    "V_RMSE" : config_types_rmses["Total"]["stresses"]
}

x = _setup()

t0 = time()
config_types_rmses, conf_raw_data = test_calc(get_model(active_model_name), x)
t1 = time() - t0

data["TT_Runtime"] = t1
add_info(active_model_name, data)