# %%
import json
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor

from generate_dataset import create_audio_generator, generate_training_set


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
    return np.array(time_shifts)

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), 'valid') / w

def score(kmeans, X, y):
    r = 0
    for i,x in enumerate(X):
        p = kmeans.predict(x.reshape(1, x.shape[0]))
        if p[0] == y[i]: r+= 1
    return r, r/len(y)

def score_regression(clf, X, y):
    errs = []
    for i, x in enumerate(X):
        p = clf.predict(x.reshape(-1, x.shape[0]))
        errs.append(p-y[i])
    return errs

# %%
# Create dataset
distance_x = (19.051e-3)/2  # Distance between hydrophones in m
distance_y = (18.37e-3)/2
coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                [distance_x, 0, -distance_y],
                [distance_x, -8.64e-3, distance_y],
                [-distance_x, -0.07e-3, distance_y]
            ))

fs = 192000
num_samples = 256
audio_generator = create_audio_generator(coord, fs, num_samples)

angles_range = (0, 180)
training_size = 10000
_X, y = generate_training_set(training_size, audio_generator, f=12500, angles_range=angles_range)

validation_size = training_size // 10
_X_val, y_val = generate_training_set(validation_size, audio_generator, f=12500, angles_range=angles_range)

# X = [np.array([moving_average(x, 10) for x in _x.T]).flatten() for _x in _X]
# X_val = [np.array([moving_average(x, 10) for x in _x.T]).flatten() for _x in _X_val]

X = [get_phase_shift(x) for x in _X]
X_val = [get_phase_shift(x) for x in _X_val]

az = [angles[0] for angles in y]
az_val = [angles[0] for angles in y_val]
# %%
# Instantiate regressors and get real data for validation

with open("Data/data.json") as f:
    data = json.load(f)

# azimuth(rad), elevation(rad), frequency(kHz)
gab_030719 = {
    "2": (90, 90, 25),"3": (120, 90, 25),"4": (150, 90, 25),"6": (60, 90),"7": (30, 90, 25),
    "8": (30, 90),"9": (60, 90),"10": (90, 90),"11": (120, 90),"12": (150, 90),
    "13": (150, 120),"14": (120, 120),"15": (90, 120),"16": (60, 120),"17": (30, 120),
    "18": (30, 120, 25), "19": (60, 120, 25), "20": (90, 120, 25), "21": (120, 120, 25), "22": (150, 120, 25),
}

gab_110118 = {
    "1": (90, 90, 25),"2": (120, 90, 25),"3": (135, 90, 25),"4": (150, 90),"5": (165, 90, 25),
    "6": (60, 90),"7": (45, 90),"8": (30, 90),"9": (15, 90),"10": (15, 90),
    "11": (180, 90), "12": (90, 100), "13": (90,120), "14": (90, 135),"15": (120, 135),"16": (135, 135), "17": (150, 135),
    "18": (60, 120, 25), "19": (45, 120, 25), "20": (30, 120, 25)
}

regressors = {
    "SVR": SVR(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "MLPRegressor": MLPRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SGDRegressor": SGDRegressor(max_iter=10000, tol=1e-3),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, 
                random_state=0, loss='ls'),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=1),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100)
}

# %%
predictors = [regressors["SVR"], regressors["KNeighborsRegressor"],
              regressors["RandomForestRegressor"], regressors["DecisionTreeRegressor"]]



# _outputs = [[pred.predict(x.reshape(-1, 3)) for pred in predictors] for x in X]
# outputs = [[o[0] for o in out] for out in _outputs]

# for i in range(len(outputs)):
#     print(i)
#     outputs[i].append(b.fast_aoa(_X[i]))

# stacker = RandomForestRegressor(n_estimators=100, random_state=0)
# stacker.fit(outputs, az)

# e = []
# for k in data:
#     _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
#     _ps = [[pred.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3))for pred in predictors] for x in _x]
#     ps = [[p[0] for p in _p] for _p in _ps]
#     p = stacker.predict(ps)
#     print(k, p[0] - gab_030719[k][0])
#     e.append(p[0] - gab_030719[k][0])
# plt.plot(e, "o-")
# plt.show()


_outputs = [[pred.predict(x.reshape(-1, 3)) for pred in predictors] for x in X]
outputs = [[o[0] for o in out] for out in _outputs]

for i in range(len(outputs)):
    print(i)
    outputs[i].append(b.fast_aoa(_X[i]))

stacker = RandomForestRegressor(n_estimators=100, random_state=0)
stacker.fit(outputs, az)

e = []
for k in data:
    _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
    _ps = [[pred.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3))for pred in predictors] for x in _x]
    ps = [[p[0] for p in _p] for _p in _ps]
    p = stacker.predict(ps)
    print(k, p[0] - gab_030719[k][0])
    e.append(p[0] - gab_030719[k][0])
plt.plot(e, "o-")
plt.show()
for i in range(len(predictors)):
    print(f"Training regressor {i}")
    predictors[i].fit(X, az)

outputs = [[pred.predict(x.reshape(-1, 3)) for pred in predictors] for x in X]

stacker = RandomForestRegressor(n_estimators=100, random_state=0)
stacker.fit([[o[0] for o in out] for out in outputs], az)

e = []
for k in data:
    _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
    _ps = [[pred.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3))for pred in predictors] for x in _x]
    ps = [[p[0] for p in _p] for _p in _ps]
    p = stacker.predict(ps)
    print(k, p[0] - gab_030719[k][0])
    e.append(p[0] - gab_030719[k][0])
plt.plot(e, "o-")
plt.show()

# %%
# train
for regressor in regressors:
    print(regressor)
    clf = regressors[regressor]
    clf.fit(X, az)
    errs = score_regression(clf, X_val, az_val)
    plt.plot(errs)
    plt.show()

    # e = []
    # for k in data:
    #     _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
    #     ps = [clf.predict(x.flatten().reshape(1,-1)) for x in _x]
    #     print(k, ps[0][0] - gab_030719[k][0])
    #     e.append(ps[0][0] - gab_030719[k][0])
    # plt.plot(e, "o-")
    
    e = []
    for k in data:
        _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
        ps = [clf.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3)) for x in _x]
        # if gab_030719[k][0] <= 30 or gab_030719[k][0] >= 150: continue
        print(k, ps[0][0] - gab_030719[k][0])
        e.append(ps[0][0] - gab_030719[k][0])
    plt.plot(e, "o-")
    plt.show()

# _data_set = np.array([np.array(data[str(i)]) for i in range(1, 11)])
# data_set = np.array([dt for _set in _data_set[:-1] for dt in _set])
# flat_data_set = data_set.reshape(data_set.shape[0], -1)
# %%
