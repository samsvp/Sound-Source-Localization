# %%
import os
import csv
import json
import random
import numpy as np
from sklearn.svm import SVC
from scipy.signal import spectrogram

from typing import List, Tuple

# %%
# Utility functions
def spherical2cartesian(phi, theta, r, deg=True):
    """
    Converts from spherical coords to cartesian.
    """
    phi_rad = np.deg2rad(phi) if deg else phi
    theta_rad = np.deg2rad(theta) if deg else theta
    return np.array([r * np.cos(phi_rad) * np.sin(theta_rad), 
                     r * np.sin(phi_rad) * np.sin(theta_rad),
                     r * np.cos(theta_rad)])

def read_json(file_name: str) -> List[np.ndarray]:
    """
    Reads the given json and returns a numpy array.
    """
    with open(file_name) as f:
        data = json.load(f)
    return [np.array(data[d]) for d in data]

def get_az_el(csv_file: str) -> List[Tuple[int, int]]:
    """
    Get Azimuth and Elevation from the given file.
    """
    with open(csv_file) as f:
        reader = csv.reader(f)
        el, az = zip(*[(int(row[3]), int(row[4])) 
            for i, row in enumerate(reader) if i != 0])
    return el, az


# %%
# constants
fs = 48000

# the original data is given through spherical coords
# so we translate then to cartesian
# double check the first angle is indeed the azimuth 
r = 4.2 * 10**-2
mic_array = np.array([
        spherical2cartesian(45, 35, r),
        spherical2cartesian(-45, -35, r),
        spherical2cartesian(135, -35, r),
        spherical2cartesian(-135, 35, r),
    ])

n_points = 10000 # number of points to get

# %%
training_size = 2000
validation_size = 500

csv_files = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev") if f.endswith(".csv")]
json_files = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev") if f.endswith(".json")]

# %%
def train(clf, json_data, csv_data):
    trained = 0
    while trained <= training_size:
        print(f"trained on {trained} signals.")
        
        i = random.randint(0, len(json_data) - 1) # choose a random file
        json_name = json_data.pop(i)
        csv_name = csv_data.pop(i)

        els, azs = get_az_el(csv_name)
        signals = read_json(json_name)
        
        _, _, _Sxxs = zip(*[spectrogram(signal[:10000,:], fs=fs, axis=0) for signal in signals])
        Sxxs = np.array([s.flatten() for s in _Sxxs])

        try:
            clf.fit(Sxxs, azs) 
            trained += len(signals)
        except Exception as e:
            print("\nError")
            print(e)
            print(Sxxs.shape)
            print(csv_name, json_name)
            print("\n")

    return clf, json_data, csv_data


clf = SVC()
clf, val_set, val_set_answers = train(clf, json_files, csv_files)

# %%
for i in range(10):
    print(f"Testing on {val_set[i]}")
    els, azs = get_az_el(val_set_answers[i])
    signals = read_json(val_set[i])
    _, _, _Sxxs = zip(*[spectrogram(signal[:10000,:], fs=fs, axis=0) for signal in signals])
    Sxxs = np.array([s.flatten() for s in _Sxxs])
    try:
        clf.fit(Sxxs, azs) 
    except Exception as e:
        print("Error")
        print(e)
        print("\n")
    clf.score()
# %%
