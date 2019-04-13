# This Function performs the actual calculation to predict values.

import copy
import warnings

import numpy as np

from .__kmeans_cluster import _cluster_labels
from .__neighbor import _neighbour_index, _neighbours

'''
Parameters : 
            dataset : pandas.series
                The data to perform prediction.
                
            n_ahead : int
                The number of values to predict.
                
            cycle : int
                Frequency of time series.
                
            k : int
                Number of clusters.
                
            w : int
                Size of window. 
                
Returns : 
            temp[-n_ahead:] : list
                List of predicted values.
            
            
'''


def _psf_predict(dataset, n_ahead, cycle, k, w):
    global clusters
    # coping original data to temp
    temp = list(copy.deepcopy(dataset))
    # print(temp)
    # print(len(temp))

    n_ahead_cycles = int(n_ahead / cycle)  # Assuming n_ahead_cycle >= 1
    n = 1
    cw = w

    while n <= n_ahead_cycles:
        # print(f'n = {n}\n')
        # Step 1. Dataset clustering (if window size was not reduced).
        if cw == w:
            clusters = _cluster_labels(temp, k)
            # print(f'cluster=\n{clusters}')
        # Step 2. Take the cluster pattern of the test.
        pattern = clusters[-cw:]

        # Step 3. Find the pattern in training data (neighbors).
        neighbors = _neighbours(clusters, cw)
        neighbor_index = _neighbour_index(clusters, cw)
        # print(f'n_i = \n{neighbor_index}')

        # Step 4. Check for patterns found.
        if not neighbor_index:
            # If no patterns were found, decrease the window size.
            # print(f'cw={cw}')
            cw = cw - 1
            # print(f'cw={cw}')
            if cw == 0:
                # If any window size produce neighbors, use the last training instance as the prediction.
                temp.append(float(dataset[-1:]))
                # Set the current window to its initial value and take next horizon.
                cw = w
                n = n + 1

                def format_warning(message, category, filename, lineno, line=''):
                    return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' + str(message) + '\n'

                warnings.formatwarning = format_warning
                warn_str = "No pattern were found in training for any window size.\nUsing last training as the prediction!"
                warnings.warn(warn_str)
        else:
            # If some patterns were found.

            # Step 5. Assess the average of the neighbors classes.
            pred = np.mean([temp[x] for x in neighbor_index])
            # print(f'data at loc = \n{[temp[x] for x in neighbor_index]}')
            # print(temp[-1:])
            # print(f'pred = {pred}\n\n')

            # Step 6. Append prediction to produce the following ones.
            temp.append(pred)

            # Step 7. Set the current window to its initial value and take next horizon.
            cw = w
            n = n + 1

    return temp[-n_ahead_cycles:]
