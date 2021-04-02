
#%%
import os
import csv
import pickle
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from beamforming import Bf
import visual_beamforming as vbf

#dataset_2019 = [f"2019/mic_dev/{f}" for f in os.listdir("2019/mic_dev")]
#%%
def trigger(y: np.ndarray, size=128, tol=0.0015, pt=50) -> np.ndarray:
    dataset = [y[(i - 1) * size:i * size] for i in range(1, y.shape[0], size)
        if sum(np.abs(y[(i - 1) * size:i * size, 0]) > tol) > pt]
    return dataset

def time_to_index_2019(filepath: str, fs: int) -> List[Tuple[int, int]]:
    dt = 1 / fs
    with open(filepath) as f:
        reader = csv.reader(f)
        indexes = [(int(float(row[1]) // dt), int(float(row[2]) // dt)) 
            for i, row in enumerate(reader) if i != 0]
    return indexes

def get_csv_file(wavfile: str) -> str:
    name_split = wavfile.split("/")
    year = name_split[0] 
    tp = name_split[1].split("_")[-1]
    fl = name_split[-1].split(".")[0]
    csvfile = f"{year}/metadata_{tp}/{fl}.csv"
    return csvfile

#%%
def create_jsons():
    for j, dt_2019 in enumerate(dataset_2019):
        print(f"{j} out of {len(dataset_2019)} processed")
        y, fs = sf.read(dt_2019)
        csv_dt_2019 = get_csv_file(dt_2019)
        indexes = time_to_index_2019(csv_dt_2019, fs)
        sounds = {i: y[idx[0]:idx[1], :].tolist() for i, idx in enumerate(indexes)}

        file_name = f"{csv_dt_2019.split('.')[0]}.json"
        with open(file_name, "w") as f:
            json.dump(sounds, f, indent=4)

    print("All data has been processed.")
# %%
def get_jsons(n: int) -> Dict[str, List[int]]:
    jsons_2019 = [f"2019_data/{f}" for f in os.listdir("2019_data/") if f.endswith("pickle")]
    jsons_2019.sort()
    data = {}
    
    for i in range(n):
        with open(jsons_2019[i],"rb") as f:
            print(jsons_2019[i])
            data[jsons_2019[i]] = pickle.load(f)
        print(f"{jsons_2019[i]} processed")
    return data
# %%
_data = get_jsons(40)
data = {key.split("/")[-1].split(".")[0]:value for key, value in _data.items()}

# %%
gab = {}
csv_2019 = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev/") if f.endswith("csv")]
csv_2019.sort()
for csv_file in csv_2019[:50]:
    csv_name = csv_file.split("/")[-1].split(".")[0]
    gab[csv_name] = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        gab[csv_name].update({str(i-1): (180 - int(row[-2]), 90 - int(row[-3]), row[0]) 
            for i, row in enumerate(reader) if i != 0})
# %%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

coord = np.array(([-0.02432757, -0.02432757, -0.02409021],
       [-0.02432757, 0.02432757, 0.02409021],
       [ 0.02432757, -0.02432757, 0.02409021],
       [ 0.02432757, 0.02432757, -0.02409021]))
fs = 48000
num_samples = 256

b = Bf(coord, fs, num_samples)
# %%
errors = {}
results = {}
for dt in data:
    print(dt)
    for n,_x in enumerate(data[dt]):
        try:
            x = np.array([moving_average(i,100) for i in _x.T]).T
            n=str(n)
            m = np.argmax(x[:10000,0])
            angles = b.fast_faoa(x[m-128:m+128,:])
            class_name = gab[dt][n][2]
            class_error = errors.get(class_name, [])
            e = angles[0]-gab[dt][n][0], angles[1]-gab[dt][n][1]
            results[n] = {"ea":e[0], "ee":e[1], "p":angles, "g":(gab[dt][n][0],gab[dt][n][1])}
            if np.abs(e[0])>180:
                e[0] = e[0] - np.signal(e[0])*360
            class_error.append(e)
            errors[class_name] = class_error
            if np.abs(errors[class_name][-1][0]) >= 100 or np.abs(errors[class_name][-1][1]) >= 90:
                print("--------------------------------")
                print(errors[class_name][-1], class_name,n)
                print(angles)
                print(gab[dt][n])
                print("--------------------------------")
                plt.plot(x[m-512:m+512,:])
                plt.show()
        except:
            pass

for class_name, class_error in errors.items():
    plt.plot(class_error, "o-")
    plt.title(class_name)
    plt.show()
# %%
#%%
# find the error
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

keys_dt = list(data.keys())
my_data = []

my_gab  = []

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