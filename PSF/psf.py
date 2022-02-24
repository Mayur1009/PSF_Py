# Class Psf

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .__optimum import _optimum_k, _optimum_w
from .__psf_predict import _psf_predict


class Psf:
    def __init__(self, data, cycle=24, k=None, w=None):
        self.data = data
        self.k = k
        self.w = w
        self.cycle = cycle

    def predict(self, n_ahead, k_values=tuple(range(3, 11)), w_values=tuple(range(5, 21))):
        # Check integrity of data (both its size and n.ahead must be multiple of cycle).
        if pd.Series.isna(self.data).any():
            raise RuntimeError('\nTime Series contain NA.')

        fit = len(self.data) % self.cycle
        if fit > 0:
            dat = list(self.data)

            def format_warning(message, category, filename, lineno, line=''):
                return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' + str(message) + '\n'

            warnings.formatwarning = format_warning
            warn_str = f"\nTime Series length is not multiple of {self.cycle}. Cutting last {fit} values!"
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
        dmin = min(self.data)
        dmax = max(self.data)
        norm_data = (self.data - dmin) / (dmax - dmin)

        # Find optimal number (K) of clusters (or use the value specified by the user).
        if self.k is None:
            self.k = _optimum_k(self.data, k_values)

        # Find optimal window size (W) (or use the value specified by the user).
        if self.w is None:
            self.w = _optimum_w(self.data, n_ahead, self.k, self.cycle, w_values)

        # Step 7. Predict the 'n.ahead' next values for the time series.
        self.preds = _psf_predict(dataset=norm_data, n_ahead=n_ahead * self.cycle, cycle=self.cycle, k=self.k, w=self.w)
        self.preds = np.array(self.preds)

        # Step 8. Denormalize predicted data.
        self.preds = self.preds * (dmax - dmin) + dmin

        return self.preds

    def model(self):
        return self

    def model_print(self):
        print('\nOriginal time-series : \n', self.data)
        print('\nPredicted Values : \n', self.preds)
        print('\nk = ', self.k)
        print('\nw = ', self.w)
        print('\ncycle = ', self.cycle)

    @staticmethod
    def psf_plot(a, b):
        new_index = []
        for i in range(len(b)):
            new_index.append(len(a) + i)
        # print(new_index)
        pred = pd.Series(data=list(b), index=new_index)
        plt.plot(a, 'k:', marker='.')
        plt.plot(pred, 'r:', marker='.')
        plt.rcParams.update({'font.size' : 20})
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend(('Original', 'Prediction'), loc='best')
        plt.show()
