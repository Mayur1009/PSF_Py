import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ._psf_predict import _psf_predict

Series = pd.Series


def _optimum_w(data, k, cycle, w_values):
    """Calculates Optimum size of window in case user didn't give as parameter.

    Parameters
    ----------
    data : list or Series (1D-array)
        A univariant time series on which prediction is performed.
    n_ahead : int
        The number of values to predict.
    k : int
        Number of clusters to form during clustering of data.
    cycle : int
        Frequency of time series.
    w_values : tuple
        A tuple of values of window size for finding optimum value of w.

    Returns
    -------
    best_w : int
        An integer from k=w_values for which prediction error is minimum.
    """
    global best_w

    # Step 1. Take validation test out from training.
    test = np.array(data[-cycle:])
    training = data[:len(data) - cycle]
    n = len(training)

    # Step 2. Find the window size (W) that minimizes the error.
    min_err = np.Inf
    for w in w_values:
        if 0 < w < n:
            # 2.1 Perform prediction with the current 'w' value.
            pred = _psf_predict(dataset=training, k=k, w=w, cycle=cycle, n_ahead=cycle * cycle)
            pred = np.array(pred)

            # 2.2 Evaluate error and update the minimum.
            err = mean_absolute_error(test, pred)

            # print(f'w = {w}\n pred = {pred}\n err = {err}\n')
            if err < min_err:
                min_err = err
                best_w = w
    return best_w
