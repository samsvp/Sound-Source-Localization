#%%
import json
import numpy as np
from time import time
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

def score(kmeans, X, y):
    r = 0
    for i,x in enumerate(X):
        p = kmeans.predict(x.reshape(1, x.shape[0]))
        print(p, y[i])
        if p[0] == y[i]: r+= 1
    return r, r/len(y)

with open("data_ipqm.json") as f:
    data = json.load(f)

_data_set = np.array([np.array(data[str(i)]) for i in range(1, 11)])
data_set = np.array([dt for _set in _data_set[:-1] for dt in _set])
flat_data_set = data_set.reshape(data_set.shape[0], -1)

y = []
for i in range(_data_set.shape[0] - 1):
    y += [i] * _data_set[i].shape[0]
y = np.array(y)

# kmeans = KMeans(init="k-means++", n_clusters=9, n_init=9, random_state=0)
# kmeans.fit(flat_data_set, y)
# s1 = score(kmeans, flat_data_set, y)

# brc = Birch(n_clusters=9)
# brc.fit(flat_data_set)
# s2 = score(brc, flat_data_set, y)

# nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree')
# nbrs.fit(flat_data_set, y)
# distances, indices = nbrs.kneighbors(flat_data_set)

clf = svm.SVC()
clf.fit(flat_data_set, y)
s = score(clf, flat_data_set, y)

# %%
# kmeans = KMeans(init="init++", n_clusters=9, n_init=4, random_state=0)
