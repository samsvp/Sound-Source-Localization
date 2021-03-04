# %%
import os
import csv
import json
import random
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from scipy.signal import spectrogram, correlate

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

def regularize_data(dataset: List[np.ndarray], n=1000) -> np.ndarray:
    """
    Regularizes the dataset and takes the first n points
    of each data
    """
    _data = np.array([dt[:1000] - dt[:1000].mean() for dt in dataset])
    data = _data / _data.std(axis=0)
    return data

def get_phase_shift(signal: np.ndarray) -> np.ndarray:
    """
    Finds the time shift between the given data.
    signal must be a 2D matrix
    """
    time_shifts = []
    t = 1 / fs
    n = len(signal[:,0])
    for i in range(1, signal.shape[-1]):
        xcorr = correlate(signal[:,0], signal[:,i])
        dt = np.linspace(-n*t, n*t, len(xcorr))
        time_shifts.append(dt[xcorr.argmax()])
    return time_shifts

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

n_points = 1000 # number of points to get

# %%
csv_files = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev") if f.endswith(".csv")]
json_files = [f"2019/metadata_dev/{f}" for f in os.listdir("2019/metadata_dev") if f.endswith(".json")]

training_size = int(0.8 * len(csv_files))
validation_size = int(0.8 * len(csv_files))

# %%
def train_p(clf):
    json_data = list(json_files)
    csv_data = list(csv_files)

    data_got = 0
    X = []
    y = []
    while data_got <= training_size:
        i = random.randint(0, len(json_data) - 1) # choose a random file
        json_name = json_data.pop(i)
        csv_name = csv_data.pop(i)

        els, azs = get_az_el(csv_name)
        signals = read_json(json_name)

        regularized_signals = regularize_data(signals)
        phase_shifts = [get_phase_shift(signal) for signal in regularized_signals]
        
        X.append(phase_shifts)
        y.append(azs)

        data_got += len(signals)
        print(f"Gotten {data_got} delays.")
    
    X = [x for _x in X for x in _x]
    y = [i for _y in y for i in _y]
    clf.fit(X, y)
    return clf, json_data, csv_data, X, y

def score_p(clf, val_set, val_set_answers):
    for i in range(10):
        print(f"Testing on {val_set[i]}")
        els, azs = get_az_el(val_set_answers[i])
        signals = read_json(val_set[i])
        regularized_signals = regularize_data(signals)
        phase_shifts = [get_phase_shift(signal) for signal in regularized_signals]

        try:
            print(clf.predict(phase_shifts), azs)
            print(clf.score(phase_shifts, azs) * 100)
        except Exception as e:
            print("Error")
            print(e)
            print("\n")

def train(clf):
    json_data = list(json_files)
    csv_data = list(csv_files)

    trained = 0
    while trained <= training_size:
        i = random.randint(0, len(json_data) - 1) # choose a random file
        json_name = json_data.pop(i)
        csv_name = csv_data.pop(i)

        els, azs = get_az_el(csv_name)
        signals = read_json(json_name)

        # remember to get the norm and throw away the frequency and timing parts
        _, _, _Sxxs = zip(*[spectrogram(signal[:10000,:] / signal[:10000,:].max(), fs=fs, axis=0) for signal in signals])
        Sxxs = np.array([s.sum(axis=1).flatten() for s in _Sxxs])
        try:
            clf.fit(Sxxs, azs) 
            trained += len(signals)
            print(f"trained on {trained} signals.")
        except Exception as e:
            print("\nError")
            print(e)
            print(Sxxs.shape)
            print(csv_name, json_name)
            print("\n")

    return clf, json_data, csv_data,

def score(clf, val_set, val_set_answers):
    for i in range(10):
        print(f"Testing on {val_set[i]}")
        els, azs = get_az_el(val_set_answers[i])
        signals = read_json(val_set[i])
        _, _, _Sxxs = zip(*[spectrogram(signal[:10000,:], fs=fs, axis=0) for signal in signals])
        Sxxs = np.array([s.sum(axis=1).flatten() for s in _Sxxs])
        try:
            print(clf.predict(Sxxs), azs)
            print(clf.score(Sxxs, azs) * 100)
        except Exception as e:
            print("Error")
            print(e)
            print("\n")


# %%
clf = DecisionTreeRegressor()
clf, val_set, val_set_answers, X, y = train_p(clf)
print("Decision Tree Scores")
score_p(clf, val_set, val_set_answers)

# %%
clf = SVR(kernel="poly", degree=5)
clf, val_set, val_set_answers = train_p(clf)
print("SVR Scores")
score_p(clf, val_set, val_set_answers)
#score_p(clf, val_set, val_set_answers)
#clf, val_set, val_set_answers = train(clf)

# %%
clf = MLPRegressor()
clf, val_set, val_set_answers = train_p(clf)
print("MLP Scores")
score_p(clf, val_set, val_set_answers)


# %%
