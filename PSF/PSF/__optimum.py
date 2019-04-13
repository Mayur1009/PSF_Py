# Function to Calculate optimum values of clusters and window.

import numpy as np
from sklearn.metrics import silhouette_score, mean_absolute_error

from .__kmeans_cluster import _cluster_labels
from .__psf_predict import _psf_predict

'''
Parameters: 
            dataset : pandas.series
                The time series.
            
            k_values : tuple
                Values of k to test with.
                 
Returns:
            best_k: int
                Optimum Number of clusters for PSF.
'''


def _optimum_k(dataset, k_values):
    global best_k
    dataset = np.array(dataset).reshape(-1, 1)

    # Find best number of clusters.
    best_s = -1
    for k in k_values:
        if 1 < k < len(dataset):
            # Using algorithm kMeans for clustering.
            clusters = _cluster_labels(dataset, k)

            # Evaluate clustering using silhouette index.
            s = silhouette_score(dataset, clusters)

            # Store best k value so far.
            if s > best_s:
                best_s = s
                best_k = k
    return best_k


'''
Parameters: 
            data : pandas.series
                The time series.
                
            n_ahead : int
                The number of values to predict.
                
            cycle : int
                Frequency of time series.
                
            k : int
                Number of clusters.
                
            w_values : tuple
                Size of window. 

Returns:
            best_w: int
                Optimum size of window for PSF.
'''


def _optimum_w(data, n_ahead, k, cycle, w_values):
    global best_w

    # Step 1. Take validation test out from training.
    test = np.array(data[-n_ahead:])
    training = data[:len(data) - n_ahead]
    n = len(training)

    # Step 2. Find the window size (W) that minimizes the error.
    min_err = np.Inf
    for w in w_values:
        if 0 < w < n:
            # 2.1 Perform prediction with the current 'w' value.
            pred = _psf_predict(dataset=training, k=k, w=w, cycle=cycle, n_ahead=cycle * n_ahead)
            pred = np.array(pred)

            # 2.2 Evaluate error and update the minimum.
            err = mean_absolute_error(test, pred)

            # print(f'w = {w}\n pred = {pred}\n err = {err}\n')
            if err < min_err:
                min_err = err
                best_w = w
    return best_w
