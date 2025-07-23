from ase.io import read
import pandas


dataset = read("../Dataset/InP_Dataset.xyz", index=":")

ds_info = {}


Nstruct = r"$N_\text{Structures}$"
Nats = r"$N_\text{Atoms}$"
Nforce = r"$N_\text{Forces}$"
Nstress = r"$N_\text{Stresses}$"

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
            "Configuration Type" : config_type.replace("_", "\_"),
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
df.at["Total", "Configuration Type"] =  "Total"

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

colformat = "r|" + 4 * r"p{0.12\linewidth}"
colformat = r"@{}r|rrrr@{}"

print(df.to_latex(index=False, column_format=colformat, formatters=formatters, multirow=True).replace(str1, str2))