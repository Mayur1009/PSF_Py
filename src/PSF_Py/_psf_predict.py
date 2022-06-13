import copy
import warnings

import numpy as np
import pandas as pd

from ._other_functions import _cluster_labels, _neighbour_index, _neighbours

Series = pd.Series


def _psf_predict(dataset, n_ahead, cycle, k, w):
    """Performs prediction on given timeseries according to Psf algorithm.

    Parameters
    ----------
    dataset : list or Series (1D-array)
        A univariant time series on which prediction is performed.
    n_ahead : int
        The number of values to predict.
    cycle : int
        Frequency of time series.
    k : int
        Number of clusters to form during clustering of data.
    w : int
        Size of window.

    Returns
    -------
    predictions : list
        List of predicted values

    """
    global clusters
    # coping original data to temp
    temp = list(copy.deepcopy(dataset))
    n_ahead_cycles = int(n_ahead / cycle)  # Assuming n_ahead_cycle >= 1
    n = 1
    cw = w

    while n <= n_ahead_cycles:

        # Step 1. Dataset clustering (if window size was not reduced).
        if cw == w:
            clusters = _cluster_labels(temp, k)

        # Step 2. Take the cluster pattern of the test.
        pattern = clusters[-cw:]

        # Step 3. Find the pattern in training data (neighbors).
        neighbors = _neighbours(clusters, cw)
        neighbor_index = _neighbour_index(clusters, cw)

        # Step 4. Check for patterns found.
        if not neighbor_index:
            # If no patterns were found, decrease the window size.
            cw = cw - 1

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

            # Step 6. Append prediction to produce the following ones.
            temp.append(pred)

            # Step 7. Set the current window to its initial value and take next horizon.
            cw = w
            n = n + 1
    predictions = temp[-n_ahead_cycles:]
    return predictions
