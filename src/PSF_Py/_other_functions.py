from re import match

import numpy as np
from sklearn.cluster import KMeans


def _cluster_labels(data, n_clusters):
    x = np.array(data).reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=1, max_iter=10, init='random').fit(x)
    cluster_labels = km.labels_
    return cluster_labels


def _neighbours(data, w):
    t = ''.join(str(i) for i in data)
    pattern = data[-w:]
    p = ''.join(str(i) for i in pattern)
    neighbour = []
    while len(t) is not 0:
        if match(p, t) and len(t) > w:
            neighbour.append(int(t[w]))
            t = t[w:]
        else:
            t = t[1:]
    return neighbour


def _neighbour_index(data, w):
    t = ''.join(str(i) for i in data)
    pattern = data[-w:]
    p = ''.join(str(i) for i in pattern)
    n_i = []
    i = 0
    while len(t) is not 0:
        if match(p, t) and len(t) > w:
            i += w
            n_i.append(i)
            t = t[w:]
        else:
            t = t[1:]
            i += 1
    return n_i
