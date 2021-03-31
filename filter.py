# %%
import json
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, \
    AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
import stacking as stk
import beamforming as bf
from utils import gab_110118, gab_030719, data, data_ipqm, plot_error, choose_coord
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
        if p[0] == y[i]: r += 1
    return r, r/len(y)

def score_regression(clf, X, y):
    errs = []
    for i, x in enumerate(X):
        p = clf.predict(x.reshape(-1, x.shape[0]))
        errs.append(p-y[i])
    return errs


# %%
coord = choose_coord(ipqm=False)

fs = 192000
num_samples = 256
b = bf.Bf(coord, fs, num_samples // 2)
audio_generator = create_audio_generator(coord, fs, num_samples)

angles_range = (0, 180)
training_size = 25000
_X, y = generate_training_set(training_size, audio_generator, f=5000, add_noise=False, angles_range=angles_range)

validation_size = training_size // 10
_X_val, y_val = generate_training_set(validation_size, audio_generator, f=5000, add_noise=False, angles_range=angles_range)

# X = [np.array([moving_average(x, 10) for x in _x.T]).flatten() for _x in _X]
# X_val = [np.array([moving_average(x, 10) for x in _x.T]).flatten() for _x in _X_val]

X = [get_phase_shift(x) for x in _X]
X_val = [get_phase_shift(x) for x in _X_val]

az, el = zip(*[(angles[0], angles[1]) for angles in y])
az_val, el_val = zip(*[(angles[0], angles[1]) for angles in y_val])

azc = [angles[0]//10 * 10 for angles in y]
azc_val = [angles[0]//10 * 10 for angles in y_val]

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

classifiers = {
    "SVC": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "MLPClassifier": MLPClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SGDClassifier": SGDClassifier(max_iter=10000, tol=1e-3),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=1),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=0, n_estimators=100)
}

# %%
# train regressors in stacking
for regressor in regressors:
    print(regressor)
    clf = regressors[regressor]
    result = stk.stacker(np.array(X),np.array(az),X_val,clf)
    errs = []
    for i in range(len(result)):
        errs.append(result[i]-az_val[i])
    plt.plot(errs)
    plt.show()

    e = []
    for k in data:
        _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
        ps = [clf.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3)) for x in _x]
        print(k, int(ps[0][0]), gab_030719[k][0], ps[0][0] - gab_030719[k][0])
        e.append(ps[0][0] - gab_030719[k][0])
    plt.plot(e, "o-")
    plt.show()

# %%
# train regressors
for regressor in regressors:
    print(regressor)
    clf = regressors[regressor]
    clf.fit(X, az)
    errs = score_regression(clf, X_val, az_val)
    plt.plot(errs)
    plt.show()

    e = []
    for k in data:
        _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
        ps = [clf.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3)) for x in _x]
        print(k, int(ps[0][0]), gab_030719[k][0], ps[0][0] - gab_030719[k][0])
        e.append(ps[0][0] - gab_030719[k][0])
    plt.plot(e, "o-")
    plt.show()

# %%
# train classifiers
for classifier in classifiers:
    print(classifier)
    clf = classifiers[classifier]
    clf.fit(X, azc)
    errs = score_regression(clf, X_val, azc_val)
    plt.plot(errs)
    plt.show()

    e = []
    for k in data:
        _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
        ps = [clf.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3)) for x in _x]
        print(k, int(ps[0][0]), gab_030719[k][0], ps[0][0] - gab_030719[k][0])
        e.append(ps[0][0] - gab_030719[k][0])
    plt.plot(e, "o-")
    plt.show()

#%%
e = []
for i in range(len(_X_val)):
    a = b.fast_aoa(_X_val[i])[0]
    e.append(a - az_val[i])
plt.plot(e, "-")
plt.show()

# %%
predictors = [regressors["KNeighborsRegressor"],regressors["RandomForestRegressor"],
              regressors["SGDRegressor"]]

for i in range(len(predictors)):
    predictors[i].fit(X, az)

outputs = [[pred.predict(x.reshape(-1, 3))[0] for pred in predictors] for x in X]
for i in range(len(outputs)):
    if not i%100: print(f"Processed {i}")
    outputs[i].append(b.fast_aoa(_X[i])[0])

# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1/np.max(X)),
        layers.Dense(128, activation='relu', input_shape=[3]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(181)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=['accuracy'])
    return model

model = build_model()

for i in range(1):
    EPOCHS = 100

    history = model.fit(np.array(X), np.array(az),
    epochs=EPOCHS, validation_split=0.2, verbose=1)

# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def build_model_b():
    model = keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1./181),
        layers.Dense(128, activation='relu', input_shape=[len(outputs[0])]),
        layers.Dense(64, activation='relu'),
        layers.Dense(181)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=['accuracy'])
    return model

model = build_model_b()

for i in range(1):
    EPOCHS = 100
    history = model.fit(np.array(outputs), np.array(az),
    epochs=EPOCHS, validation_split=0.2, verbose=1)
    e = []
    eb = []
    for k in data:
        _x = [np.array([moving_average(d, 10) for d in np.array(dt).T]) for dt in data[k]]
        _ps = [[
            pred.predict(get_phase_shift(np.array(x.T)).reshape(-1, 3))[0] 
        for pred in predictors] for x in _x]
        for j in range(len(_ps)):
            _ps[j].append(b.fast_faoa(np.array(data[k][j]))[0])
        ps = model.predict(np.array(_ps))
        print(k, np.argmax(ps[0]), gab_030719[k][0], np.argmax(ps[0]) - gab_030719[k][0])
        eb.append(np.abs(b.fast_faoa(np.array(data[k][0]))[0] - gab_030719[k][0]))
        e.append(np.abs(np.argmax(ps[0]) - gab_030719[k][0]))
    plt.plot(e, "o-b")
    plt.plot(eb, "o-r")
    plt.title(i)
    plt.savefig(f"{i}_b.png")
    plt.clf()
    model.save(f"weights/model_{i}")