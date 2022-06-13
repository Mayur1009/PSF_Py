# Imports
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._optimum_k import _optimum_k
from ._optimum_w import _optimum_w
from ._psf_predict import _psf_predict

Series = pd.Series

__all__ = ['Psf', 'psf_plot']


# Psf class for Psf model creation
class Psf:
    """
    A class for creating Psf model for forecasting future values and to perform prediction on the created model.
    """

    def __init__(self, data, cycle=24, k=None, w=None):
        """
        Parameters
        ----------
        data : list or Series
            A univariant time series on which prediction is performed.
        cycle : int
            Frequency of time series.
        k : int
            Number of clusters to form during clustering of data.
        w : int
            Size of window.
        """
        self.data = data
        self.k = k
        self.w = w
        self.cycle = cycle
        self.preds = []
        self.dmax = max(self.data)
        self.dmin = min(self.data)

    def predict(self, n_ahead, k_values=tuple(range(2, 11)), w_values=tuple(range(5, 21))):
        """
        Performs prediction on the Psf model created using the class Psf.

        Parameters
        ----------
        n_ahead : int
            The number of values to predict.
        k_values : tuple
            A tuple of values of number of cluster to perform clustering and finding optimum value of k.
        w_values : tuple
            A tuple of values of window size for finding optimum value of w.

        Returns
        -------
        preds : numpy.ndarray
            Array of predicted values.
        """
        # Check integrity of data (both its size and n.ahead must be multiple of cycle).
        if pd.Series.isna(self.data).any():
            raise RuntimeError('\nTime Series contain NA.')

        fit = len(self.data) % self.cycle
        if fit > 0:
            dat = list(self.data)

            def format_warning(message, category, filename, lineno, line=''):
                return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' + str(message) + '\n'

            warnings.formatwarning = format_warning
            warn_str = f"\nTime Series length is not multiple of {self.cycle}. Cutting first {fit} values!"
            warnings.warn(warn_str)
            dat = self.data[fit:]
            self.data = pd.Series(dat)

        original_n_ahead = n_ahead
        fit = n_ahead % self.cycle
        if fit > 0:
            n_ahead = int(self.cycle * np.ceil(n_ahead / self.cycle))

            def format_warning(message, category, filename, lineno, line=''):
                return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' + str(message) + '\n'

            warnings.formatwarning = format_warning
            warn_str = f"\nPrediction horizon is not multiple of {self.cycle}. Using {n_ahead} as prediction horizon!"
            warnings.warn(warn_str)

        #  Normalize data.
        dmin = self.dmin
        dmax = self.dmax
        norm_data = (self.data - dmin) / (dmax - dmin)

        # Find optimal number (K) of clusters (or use the value specified by the user).
        if self.k is None:
            self.k = _optimum_k(self.data, k_values)

        # Find optimal window size (W) (or use the value specified by the user).
        if self.w is None:
            self.w = _optimum_w(self.data, self.k, self.cycle, w_values)

        # Step 7. Predict the 'n.ahead' next values for the time series.
        self.preds = _psf_predict(dataset=norm_data, n_ahead=n_ahead * self.cycle, cycle=self.cycle, k=self.k, w=self.w)
        self.preds = np.array(self.preds)[:original_n_ahead]

        # Step 8. Denormalize predicted data.
        self.preds = self.preds * (dmax - dmin) + dmin

        return self.preds

    def model_print(self):
        """
        Prints the model created using the class Psf, displays all Attributes of the class.
        """
        if self.k is None:
            self.k = _optimum_k(self.data, k_values=tuple(range(2, 11)))

        if self.w is None:
            self.w = _optimum_w(self.data, k=self.k, cycle=self.cycle, w_values=tuple(range(5, 21)))

        params = vars(self)

        print('\nOriginal time-series : \n', params['data'])

        print('\nk = ', params['k'])

        print('\nw = ', params['w'])

        print('\ncycle = ', params['cycle'])

        print('\ndmin = ', params['dmin'])

        print('\ndmax = ', params['dmax'])

        print('\ntype = ', type(self))


def psf_plot(a, b):
    """
    Plots the PSF model and predicted values.

    Parameters
    ----------
    a : Psf
        PSF model created using Psf class.

    b : pandas.Series or list or numpy.ndarray
        Predicted values

    """
    # retrieve original-data from Psf model
    x = a.data

    # change the index of original data to range from 0 to len(original data).
    new_index = []
    i = 0
    while i < len(x):
        new_index.append(i)
        i = i + 1
    x = pd.Series(list(x), index=new_index)

    # change the index of predictions to start from len(original data).
    new_index = []
    for i in range(len(b)):
        new_index.append(len(x) + i)
    pred = pd.Series(data=list(b), index=new_index)

    # change the default aesthetics of matplotlib plot
    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (15, 15),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large'
              }
    plt.rcParams.update(params)

    # plot the original data with black colour and dotted line
    plt.plot(x, 'k-', marker='.', markersize=7.5)

    # plot the predictions with red colour and dotted line
    plt.plot(pred, 'r-', marker='.', markersize=7.5)

    # change the font size to 12 and label x axis as 'Time' and y axis as 'Values'.

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(('Original', 'Prediction'), loc='upper right')
    plt.show()
