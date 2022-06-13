"""
Package: PSF_Py
---------------

The Algorithm Pattern Sequence based Forecasting (PSF) was first proposed by Martinez Alvarez, et al., 2008 and then modified and suggested improvement by Martinez Alvarez, et al., 2011. The technical detailes are mentioned in referenced articles. PSF algorithm consists of various statistical operations like:

* Data Normalization/ Denormalization
* Calculation of optimum Window size (W)
* Calculation of optimum cluster size (k)
* Pattern Sequence based Forecasting
* RMSE/MAE Calculation, etc...

This module consist of a class, Psf, which is used to create PSF model and perform forecasting.

Example
-------
********

    From the PSF_Py package import Psf, get_ts, psf_plot

    >>> from PSF_Py import Psf,get_ts,psf_plot

    Access the "nottem" time series provided in the package

    >>> ts = get_ts("nottem")

    Create a PSF model which is then used for prediction

    >>> a = Psf(ts, cycle=12)

    Display the PSF model

    >>> a.model_print()
        Original time-series :
         0      40.6
        1      40.8
        2      44.4
        3      46.7
        4      54.1
        5      58.5
        6      57.7
        7      56.4
        8      54.3
        9      50.5
        10     42.9
        11     39.8
        12     44.2
        13     39.8
        14     45.1
        15     47.0
        16     54.1
        17     58.7
        18     66.3
        19     59.9
        20     57.0
        21     54.2
        22     39.7
        23     42.8
        24     37.5
        25     38.7
        26     39.5
        27     42.1
        28     55.7
        29     57.8
               ...
        210    61.4
        211    61.8
        212    56.3
        213    50.9
        214    41.4
        215    37.1
        216    42.1
        217    41.2
        218    47.3
        219    46.6
        220    52.4
        221    59.0
        222    59.6
        223    60.4
        224    57.0
        225    50.7
        226    47.8
        227    39.2
        228    39.4
        229    40.9
        230    42.4
        231    47.8
        232    52.4
        233    58.0
        234    60.7
        235    61.8
        236    58.2
        237    46.7
        238    46.6
        239    37.8
        Length: 240, dtype: float64
        k =  2
        w =  12
        cycle =  12
        dmin =  31.3
        dmax =  66.5
        type =  <class 'PSF_Py.psf.Psf'>

    Predict future n_ahead values

    >>> b = a.predict(n_ahead=12)
    >>> b
        array([39.92857143, 38.48571429, 42.37142857, 46.34285714, 52.     , 58.3       , 61.47142857, 61.65714286, 56.98571429, 49.85714286, 42.46      , 39.19      ])

    Plot the model and predicted values

    >>> psf_plot(a, b)

    The values of k and w can be accessed using

    >>> k, w = a.k, a.w
"""

from .psf import Psf, psf_plot
from .ts import get_ts
from .dpsf import Dpsf, dpsf_plot
from .plot import plot_psf

__version__ = "0.3"

__all__ = ['Psf', 'get_ts', 'psf_plot', 'Dpsf', 'dpsf_plot', 'plot_psf']

__author__ = ["Mayur Shende", "Neeraj Bokde"]
