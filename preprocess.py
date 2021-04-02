#%%
import os
import csv
import json
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple

from beamforming import Bf

dataset_2019 = [f"2019/mic_dev/{f}" for f in os.listdir("2019/mic_dev")]
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
    jsons_2019 = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev") if f.endswith("json")]
    data = {}
    for i in range(n):
        with open(jsons_2019[i]) as f:
            data[jsons_2019[i]] = json.load(f)
        print(f"{jsons_2019[i]} processed")
    return data
# %%
_data = get_jsons(5)
data = {key.split("/")[-1].split(".")[0]:value for key, value in _data.items()}

# %%
gab = {}
for csv_file in ["2019/metadata_dev/split1_ir0_ov1_1.csv",
                 "2019/metadata_dev/split1_ir0_ov1_2.csv",
                 "2019/metadata_dev/split1_ir0_ov1_3.csv",
                 "2019/metadata_dev/split1_ir0_ov1_4.csv",
                 "2019/metadata_dev/split1_ir0_ov1_5.csv"]:
    csv_name = csv_file.split("/")[-1].split(".")[0]
    gab[csv_name] = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        gab[csv_name].update({str(i-1): (180 - int(row[-2]), 90 - int(row[-3])) 
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
