# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import Dict, Any

import beamforming as bf
from utils import gab_110118, gab_030719, data, data_ipqm, plot_error, choose_coord
from generate_dataset import create_audio_generator, generate_training_set, generate_all_angles


# %%
# Create dataset
ipqm = True
coord = choose_coord(ipqm)
fs = 192000
num_samples = 256
b = bf.Bf(coord, fs, num_samples // 2)
audio_generator = create_audio_generator(coord, fs, num_samples)
X, y = [], []
for f in [5000]:
    print(f)
    _X, _y = generate_all_angles(audio_generator, f=f, add_noise=False)
    X += _X
    y += _y
az, el = zip(*[(angles[0], angles[1]) for angles in y])

#%%
# find the error
results = {}
for i in range(len(X)):
    if not (i%100): print(i)
    angles = b.fast_faoa(X[i])
    ea = (angles[0] - az[i])
    ee = (angles[1] - el[i])
    results[i] = {"ea": ea, "ee": ee, 
        "p": angles, "g": (ea, ee)}

dict_a = {}
dict_e = {}
for key, r in results.items():
    a = dict_a.get(r["p"][0], [])
    a.append(((r["p"],r["g"])))
    dict_a[r["p"][0]] = a

    e = dict_e.get(r["p"][1], [])
    e.append(((r["p"],r["g"])))
    dict_e[r["p"][1]] = e

# %%
class_width = 10
n_classes = 180 // class_width

def get_errors(dict_x: Dict[Any, Any], dict_x_cl: Dict[Any, Any], angle: int):
    _dict_x_cl = {}
    for angle_0, values in dict_x.items():
        # get the errors from the given angle
        _dict_x_cl[angle_0] = {n:[] for n in range(n_classes + 1)}
        dict_x_cl[angle_0] = {n:-1 for n in range(n_classes + 1)}
        for value in values:
            _dict_x_cl[angle_0][value[0][angle-1]//class_width].append(value[1])
        for angle_1 in _dict_x_cl[angle_0]:
            dict_x_cl[angle_0][angle_1] = [
                np.nan_to_num(np.mean([a[0] for a in _dict_x_cl[angle_0][angle_1]])),
                np.nan_to_num(np.mean([a[1] for a in _dict_x_cl[angle_0][angle_1]]))
            ]

dict_a_cl, dict_e_cl = {}, {}

get_errors(dict_a, dict_a_cl, 0)
get_errors(dict_e, dict_e_cl, 1)

my_data = data_ipqm if ipqm else data
my_gab = gab_110118 if ipqm else gab_030719

ea, ea_m = [], []
ee, ee_m = [], []

for k in my_gab:
    a, ele = b.fast_faoa(np.array(my_data[k][0]))
    print(a, ele, dict_a_cl[a][ele // class_width])
    ea.append((a - my_gab[k][0]))
    ea_m.append(a - dict_a_cl[a][ele // class_width][0] - my_gab[k][0])
    ee.append((ele - my_gab[k][1]))
    ee_m.append(ele - dict_e_cl[ele][a // class_width][1] - my_gab[k][1])

plt.plot([0]*len(ea))
plt.plot(ea, "o-b")
plt.plot(ea_m, "o-r")
plt.show()
plt.plot([0]*len(ea))
plt.plot(ee, "o-b")
plt.plot(ee_m, "o-r")

print("azimuth", np.abs(ea).sum(), "azimuth with feedback", np.abs(ea_m).sum())
print("elevation", np.abs(ee).sum(), "elevation with feedback", np.abs(ee_m).sum())

# %%

# %%