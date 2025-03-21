import json
import os

def add_info(model_name, data_dict):
    fname = f"../Saved_Data/{model_name}.json"

    if not os.path.exists(fname):
        f = open(fname, "w")
        data = {}
    else:
        with open(fname, "r") as f:
            data = json.load(f)
        f = open(fname, "w")

    data = data | data_dict

    json.dump(data, f, indent=4)
    f.close()