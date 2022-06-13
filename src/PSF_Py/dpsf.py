from .psf import Psf
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from ._optimum_k import _optimum_k
from ._optimum_w import _optimum_w


class Dpsf:
    def __init__(self, data, cycle=24, k=None, w=None):
        self.data = data
        self.cycle = cycle
        self.k = k
        self.w = w
        self.dmin = min(self.data)
        self.dmax = max(self.data)
        self.preds = []

    def predict(self, n_ahead, k_values=tuple(range(2, 11)), w_values=tuple(range(5, 21))):
        train = pd.Series(self.data)
        diff = train.diff()
        for_psf = copy.deepcopy(diff)
        for_psf[0] = diff[self.cycle]
        a = Psf(for_psf, 12, self.k, self.w)
        b = a.predict(n_ahead, k_values, w_values)
        diff = diff.append(pd.Series(b), ignore_index=True)
        tsa = np.array(train)
        undiff = np.r_[tsa[0], diff[1:]].cumsum()
        undiff = pd.Series(undiff)
        preds = undiff[-n_ahead:]
        return np.array(preds)

    def model_print(self):
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


def dpsf_plot(a: Dpsf, b) -> None:
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
    plt.legend(('Original', 'Prediction'))
    plt.show()
