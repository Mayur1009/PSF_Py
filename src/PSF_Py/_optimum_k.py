import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from PSF_Py._other_functions import _cluster_labels

Series = pd.Series


def _optimum_k(dataset, k_values):
    """Calculates Optimum number of clusters in case user didn't give as parameter.

    Parameters
    ----------
    dataset : list or Series (1D-array)
        A univariant time series on which prediction is performed.
    k_values : tuple
        A tuple of values of number of cluster to perform clustering and finding optimum value of k.

    Returns
    -------
    best_k : int
        An integer from k_values for which the silhouette_score is maximum.
    """
    global best_k
    dataset = np.array(dataset).reshape(-1, 1)

    # Find best number of clusters.
    best_s = -1
    for k in k_values:
        if 1 < k < len(dataset):
            # Using algorithm kMeans for clustering.
            clusters = _cluster_labels(dataset, k)

            # Evaluate clustering using silhouette index.
            s = round(silhouette_score(dataset, clusters), 5)

            # Store best k value so far.
            if s > best_s:
                best_s = s
                best_k = k
    return best_k
