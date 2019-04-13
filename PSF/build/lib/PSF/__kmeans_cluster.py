# This function calculates kmeans clusters given time series.

import numpy as np
from sklearn.cluster import KMeans

'''
Parameters: 
            dataset : pandas.series
                The data to perform k-means clustering on.
            
            n_clusters : int
                 Number of clusters (k) to form.
                 
Returns:
            cluster_labels : array
                Index of the cluster each sample belongs to.
'''


def _cluster_labels(dataset, n_clusters):
    # calculate clusters
    dataset = np.array(dataset).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=1).fit(dataset)
    cluster_labels = kmeans.labels_
    return cluster_labels
