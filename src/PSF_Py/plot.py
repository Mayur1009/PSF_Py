from .psf import Psf
from .dpsf import Dpsf
import pandas as pd
import matplotlib.pyplot as plt


def plot_psf(a, b):
    """
       Plots the PSF model and predicted values.

       Parameters
       ----------
       a : Psf or Dpsf
           PSF or Dpsf model created using Psf class.

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
    plt.plot(x, 'k:', marker='.')

    # plot the predictions with red colour and dotted line
    plt.plot(pred, 'r:', marker='.')

    # change the font size to 12 and label x axis as 'Time' and y axis as 'Values'.

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(('Original', 'Prediction'))
    plt.show()
