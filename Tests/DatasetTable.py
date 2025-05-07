from ase.io import read
import pandas


dataset = read("../Dataset/InP_Dataset.xyz", index=":")

ds_info = {}


Nstruct = "Number of Structures"
Nats = "Total Number of Atoms"
Nforce = "Number of Force Observations"
Nstress = "Number of Stress Observations"

for image in dataset:
    config_type = image.info["config_type"]
    N = len(image)

    try:
        image.get_stress()
        has_stress = True
    except:
        has_stress = False

    if config_type not in ds_info.keys():
        ds_info[config_type] = {
            "Config Type" : config_type.replace("_", "\_"),
            Nstruct : 1,
            Nats : 1,
            Nforce : 3 * N,
            Nstress : 9 * has_stress
        }
    else:
        ds_info[config_type][Nstruct] += 1
        ds_info[config_type][Nats] += N
        ds_info[config_type][Nforce] += 3 * N
        ds_info[config_type][Nstress] += 9 * has_stress

df = pandas.DataFrame(ds_info).T.sort_index()
df.loc['Total'] = df.sum()
df.at["Total", "Config Type"] =  "Total"

str1 = r"""\\
Total"""

str2 = r"""\\
\midrule
Total"""

formatters = {
    Nstruct : "{:,}",
    Nats : "{:,}",
    Nforce : "{:,}",
    Nstress : lambda x: "" if x==0 else f"{x:,}"
}


print(df.to_latex(index=False, column_format="l|rrrr", formatters=formatters).replace(str1, str2))