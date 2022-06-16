from pandas import Series


def get_ts(name):
    """
    Returns a pandas Series consisting of the time series of the name given as input.
    \nNote : This function is not intended to give the entire dataset(dataframe) as output. The Psf algorithm is used to forecast univariate time series, therefore, the function return only a Series(1D-array) that consist of the time series data for the particular dataset.

    Parameters
    ----------
    name : str
        The name of the timeseries to get. "name" can have the following values : ['AirPassengers', 'nottem', 'sunspots', 'penguin', 'morley', 'Nile', 'wineind','co2','gas']

    Returns
    -------
    ts : Series
        The time series. Data type of the returned time series is pandas.Series.

    """
    if name is 'AirPassengers':
        ts = airpassengers()
    elif name is 'nottem':
        ts = nottem()
    elif name is 'sunspots':
        ts = sunspots()
    elif name is 'penguin':
        ts = penguin()
    elif name is 'morley':
        ts = morley()
    elif name is 'Nile':
        ts = nile()
    elif name is 'wineind':
        ts = wineind()
    elif name is 'co2':
        ts = co2()
    elif name is 'gas':
        ts = gas()
    else:
        raise ValueError(
            'Time Series is currently unavailable in the package. Following time series are available: '
            '"AirPassengers", "nottem", "sunspots", "penguin", "morley", "Nile", "wineind","co2","gas"'
        )
    return ts


def airpassengers ():
    """
    Description
    -----------
    The classic Box & Jenkins airline data. Monthly totals of international airline passengers, 1949 to 1960.

    Format
    ------
    A monthly time series, in thousands.

    Source
    ------
    Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting of 'AirPassengers' dataset.
    """

    ts = Series(
        [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158,
         133,
         114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230,
         242,
         209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234,
         264,
         302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317,
         313,
         318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340,
         318,
         362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362,
         405,
         417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])
    return ts


def nottem ():
    """
    Description
    -----------
    A time series object containing average air temperatures at Nottingham Castle in degrees Fahrenheit for 20 years.


    Source
    ------
    Anderson, O. D. (1976) Time Series Analysis and Forecasting: The Box-Jenkins approach. Butterworths. Series R.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting of 'nottem' dataset.
    """
    ts = Series(
        [40.6, 40.8, 44.4, 46.7, 54.1, 58.5, 57.7, 56.4, 54.3, 50.5, 42.9, 39.8, 44.2, 39.8, 45.1, 47.0, 54.1, 58.7,
         66.3, 59.9, 57.0, 54.2, 39.7, 42.8, 37.5, 38.7, 39.5, 42.1, 55.7, 57.8, 56.8, 54.3, 54.3, 47.1, 41.8, 41.7,
         41.8, 40.1, 42.9, 45.8, 49.2, 52.7, 64.2, 59.6, 54.4, 49.2, 36.3, 37.6, 39.3, 37.5, 38.3, 45.5, 53.2, 57.7,
         60.8, 58.2, 56.4, 49.8, 44.4, 43.6, 40.0, 40.5, 40.8, 45.1, 53.8, 59.4, 63.5, 61.0, 53.0, 50.0, 38.1, 36.3,
         39.2, 43.4, 43.4, 48.9, 50.6, 56.8, 62.5, 62.0, 57.5, 46.7, 41.6, 39.8, 39.4, 38.5, 45.3, 47.1, 51.7, 55.0,
         60.4, 60.5, 54.7, 50.3, 42.3, 35.2, 40.8, 41.1, 42.8, 47.3, 50.9, 56.4, 62.2, 60.5, 55.4, 50.2, 43.0, 37.3,
         34.8, 31.3, 41.0, 43.9, 53.1, 56.9, 62.5, 60.3, 59.8, 49.2, 42.9, 41.9, 41.6, 37.1, 41.2, 46.9, 51.2, 60.4,
         60.1, 61.6, 57.0, 50.9, 43.0, 38.8, 37.1, 38.4, 38.4, 46.5, 53.5, 58.4, 60.6, 58.2, 53.8, 46.6, 45.5, 40.6,
         42.4, 38.4, 40.3, 44.6, 50.9, 57.0, 62.1, 63.5, 56.3, 47.3, 43.6, 41.8, 36.2, 39.3, 44.5, 48.7, 54.2, 60.8,
         65.5, 64.9, 60.1, 50.2, 42.1, 35.8, 39.4, 38.2, 40.4, 46.9, 53.4, 59.6, 66.5, 60.4, 59.2, 51.2, 42.8, 45.8,
         40.0, 42.6, 43.5, 47.1, 50.0, 60.5, 64.6, 64.0, 56.8, 48.6, 44.2, 36.4, 37.3, 35.0, 44.0, 43.9, 52.7, 58.6,
         60.0, 61.1, 58.1, 49.6, 41.6, 41.3, 40.8, 41.0, 38.4, 47.4, 54.1, 58.6, 61.4, 61.8, 56.3, 50.9, 41.4, 37.1,
         42.1, 41.2, 47.3, 46.6, 52.4, 59.0, 59.6, 60.4, 57.0, 50.7, 47.8, 39.2, 39.4, 40.9, 42.4, 47.8, 52.4, 58.0,
         60.7, 61.8, 58.2, 46.7, 46.6, 37.8])
    return ts


def penguin():
    """
    This time series has been downloaded from https://timeseries.weebly.com/uploads/2/1/0/8/21086414/_penguin.csv

    Returns
    -------
     ts : Series
        Returns a pandas series consisting of 'Number' column from dataset.
    """
    ts = Series(
        [753, 448, 356, 504, 698, 256, 361, 476, 541, 812, 914, 998, 762, 461, 374, 521, 712, 274, 384, 492, 561,
         821,
         930, 1014, 779, 478, 391, 543, 910, 287, 399, 511, 584, 843, 951, 1028, 793, 491, 407, 559, 755, 304, 418,
         536,
         601, 862, 973, 1058, 587, 281, 210, 349, 561, 123, 225, 304, 416, 642, 769, 853, 572, 273, 208, 341, 553,
         136,
         231, 299, 403, 632, 759, 848, 561, 268, 212, 331, 542, 128, 225, 301, 389, 624, 748, 842])
    return ts


def morley ():
    """
    Description
    -----------
    A classical data of Michelson (but not this one with Morley) on measurements done in 1879 on the speed of light. The data consists of five experiments, each consisting of 20 consecutive ‘runs’. The response is the speed of light measurement, suitably coded (km/sec, with 299000 subtracted).

    Format
    ------
    A data frame with 100 observations on the following 3 variables.
        Expt : The experiment number, from 1 to 5.

        Run : The run number within each experiment.

        Speed : speed-of-light measurement.

    Details
    -------
    The data is here viewed as a randomized block experiment with ‘experiment’ and ‘run’ as the factors. ‘run’ may also be considered a quantitative variate to account for linear (or polynomial) changes in the measurement over the course of a single experiment.

    Note
    ----
    This is the same dataset as michelson in package MASS.


    Source
    ------
    [1] A. J. Weekes (1986) A Genstat Primer. London: Edward Arnold.

    [2] S. M. Stigler (1977) Do robust estimators work with real data? Annals of Statistics 5, 1055–1098. (See Table 6.)

    [3] A. A. Michelson (1882) Experimental determination of the velocity of light made at the United States Naval Academy, Annapolis. Astronomic Papers 1 135–8. U.S. Nautical Almanac Office. (See Table 24.)


    Returns
    -------
    ts : Series
        Returns a pandas series consisting of 'speed' column from dataset.
    """
    ts = Series(
        [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880, 1000, 980, 930, 650, 760, 810, 1000, 1000, 960, 960,
         960,
         940, 960, 940, 880, 800, 850, 880, 900, 840, 830, 790, 810, 880, 880, 830, 800, 790, 760, 800, 880, 880,
         880,
         860, 720, 720, 620, 860, 970, 950, 880, 910, 850, 870, 840, 840, 850, 840, 840, 840, 890, 810, 810, 820,
         800,
         770, 760, 740, 750, 760, 910, 920, 890, 860, 880, 720, 840, 850, 850, 780, 890, 840, 780, 810, 760, 810,
         790,
         810, 820, 850, 870, 870, 810, 740, 810, 940, 950, 800, 810, 870])
    return ts


def sunspots ():
    """
    Description
    -----------
    Monthly mean relative sunspot numbers from 1749 to 1983. Collected at Swiss Federal Observatory, Zurich until 1960, then Tokyo Astronomical Observatory.

    Format
    ------
    A time series of monthly data from 1749 to 1983.

    Source
    ------
    Andrews, D. F. and Herzberg, A. M. (1985) Data: A Collection of Problems from Many Fields for the Student and Research Worker. New York: Springer-Verlag.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting sunspots data.
    """
    ts = Series(
        [58, 62.6, 70, 55.7, 85, 83.5, 94.8, 66.3, 75.9, 75.5, 158.6, 85.2, 73.3, 75.9, 89.2, 88.3, 90, 100, 85.4,
         103,
         91.2, 65.7, 63.3, 75.4, 70, 43.5, 45.3, 56.4, 60.7, 50.7, 66.3, 59.8, 23.5, 23.2, 28.5, 44, 35, 50, 71,
         59.3,
         59.7, 39.6, 78.4, 29.3, 27.1, 46.6, 37.6, 40, 44, 32, 45.7, 38, 36, 31.7, 22.2, 39, 28, 25, 20, 6.7, 0, 3,
         1.7,
         13.7, 20.7, 26.7, 18.8, 12.3, 8.2, 24.1, 13.2, 4.2, 10.2, 11.2, 6.8, 6.5, 0, 0, 8.6, 3.2, 17.8, 23.7, 6.8,
         20,
         12.5, 7.1, 5.4, 9.4, 12.5, 12.9, 3.6, 6.4, 11.8, 14.3, 17, 9.4, 14.1, 21.2, 26.2, 30, 38.1, 12.8, 25, 51.3,
         39.7, 32.5, 64.7, 33.5, 37.6, 52, 49, 72.3, 46.4, 45, 44, 38.7, 62.5, 37.7, 43, 43, 48.3, 44, 46.8, 47, 49,
         50,
         51, 71.3, 77.2, 59.7, 46.3, 57, 67.3, 59.5, 74.7, 58.3, 72, 48.3, 66, 75.6, 61.3, 50.6, 59.7, 61, 70, 91,
         80.7,
         71.7, 107.2, 99.3, 94.1, 91.1, 100.7, 88.7, 89.7, 46, 43.8, 72.8, 45.7, 60.2, 39.9, 77.1, 33.8, 67.7, 68.5,
         69.3, 77.8, 77.2, 56.5, 31.9, 34.2, 32.9, 32.7, 35.8, 54.2, 26.5, 68.1, 46.3, 60.9, 61.4, 59.7, 59.7, 40.2,
         34.4, 44.3, 30, 30, 30, 28.2, 28, 26, 25.7, 24, 26, 25, 22, 20.2, 20, 27, 29.7, 16, 14, 14, 13, 12, 11,
         36.6,
         6, 26.8, 3, 3.3, 4, 4.3, 5, 5.7, 19.2, 27.4, 30, 43, 32.9, 29.8, 33.3, 21.9, 40.8, 42.7, 44.1, 54.7, 53.3,
         53.5, 66.1, 46.3, 42.7, 77.7, 77.4, 52.6, 66.8, 74.8, 77.8, 90.6, 111.8, 73.9, 64.2, 64.3, 96.7, 73.6,
         94.4,
         118.6, 120.3, 148.8, 158.2, 148.1, 112, 104, 142.5, 80.1, 51, 70.1, 83.3, 109.8, 126.3, 104.4, 103.6,
         132.2,
         102.3, 36, 46.2, 46.7, 64.9, 152.7, 119.5, 67.7, 58.5, 101.4, 90, 99.7, 95.7, 100.9, 90.8, 31.1, 92.2, 38,
         57,
         77.3, 56.2, 50.5, 78.6, 61.3, 64, 54.6, 29, 51.2, 32.9, 41.1, 28.4, 27.7, 12.7, 29.3, 26.3, 40.9, 43.2,
         46.8,
         65.4, 55.7, 43.8, 51.3, 28.5, 17.5, 6.6, 7.9, 14, 17.7, 12.2, 4.4, 0, 11.6, 11.2, 3.9, 12.3, 1, 7.9, 3.2,
         5.6,
         15.1, 7.9, 21.7, 11.6, 6.3, 21.8, 11.2, 19, 1, 24.2, 16, 30, 35, 40, 45, 36.5, 39, 95.5, 80.3, 80.7, 95,
         112,
         116.2, 106.5, 146, 157.3, 177.3, 109.3, 134, 145, 238.9, 171.6, 153, 140, 171.7, 156.3, 150.3, 105, 114.7,
         165.7, 118, 145, 140, 113.7, 143, 112, 111, 124, 114, 110, 70, 98, 98, 95, 107.2, 88, 86, 86, 93.7, 77, 60,
         58.7, 98.7, 74.7, 53, 68.3, 104.7, 97.7, 73.5, 66, 51, 27.3, 67, 35.2, 54, 37.5, 37, 41, 54.3, 38, 37, 44,
         34,
         23.2, 31.5, 30, 28, 38.7, 26.7, 28.3, 23, 25.2, 32.2, 20, 18, 8, 15, 10.5, 13, 8, 11, 10, 6, 9, 6, 10, 10,
         8,
         17, 14, 6.5, 8, 9, 15.7, 20.7, 26.3, 36.3, 20, 32, 47.2, 40.2, 27.3, 37.2, 47.6, 47.7, 85.4, 92.3, 59, 83,
         89.7, 111.5, 112.3, 116, 112.7, 134.7, 106, 87.4, 127.2, 134.8, 99.2, 128, 137.2, 157.3, 157, 141.5, 174,
         138,
         129.2, 143.3, 108.5, 113, 154.2, 141.5, 136, 141, 142, 94.7, 129.5, 114, 125.3, 120, 123.3, 123.5, 120,
         117,
         103, 112, 89.7, 134, 135.5, 103, 127.5, 96.3, 94, 93, 91, 69.3, 87, 77.3, 84.3, 82, 74, 72.7, 62, 74, 77.2,
         73.7, 64.2, 71, 43, 66.5, 61.7, 67, 66, 58, 64, 63, 75.7, 62, 61, 45.8, 60, 59, 59, 57, 56, 56, 55, 55.5,
         53,
         52.3, 51, 50, 29.3, 24, 47, 44, 45.7, 45, 44, 38, 28.4, 55.7, 41.5, 41, 40, 11.1, 28.5, 67.4, 51.4, 21.4,
         39.9,
         12.6, 18.6, 31, 17.1, 12.9, 25.7, 13.5, 19.5, 25, 18, 22, 23.8, 15.7, 31.7, 21, 6.7, 26.9, 1.5, 18.4, 11,
         8.4,
         5.1, 14.4, 4.2, 4, 4, 7.3, 11.1, 4.3, 6, 5.7, 6.9, 5.8, 3, 2, 4, 12.4, 1.1, 0, 0, 0, 3, 2.4, 1.5, 12.5,
         9.9,
         1.6, 12.6, 21.7, 8.4, 8.2, 10.6, 2.1, 0, 0, 4.6, 2.7, 8.6, 6.9, 9.3, 13.9, 0, 5, 23.7, 21, 19.5, 11.5,
         12.3,
         10.5, 40.1, 27, 29, 30, 31, 32, 31.2, 35, 38.7, 33.5, 32.6, 39.8, 48.2, 47.8, 47, 40.8, 42, 44, 46, 48, 50,
         51.8, 38.5, 34.5, 50, 50, 50.8, 29.5, 25, 44.3, 36, 48.3, 34.1, 45.3, 54.3, 51, 48, 45.3, 48.3, 48, 50.6,
         33.4,
         34.8, 29.8, 43.1, 53, 62.3, 61, 60, 61, 44.1, 51.4, 37.5, 39, 40.5, 37.6, 42.7, 44.4, 29.4, 41, 38.3, 39,
         29.6,
         32.7, 27.7, 26.4, 25.6, 30, 26.3, 24, 27, 25, 24, 12, 12.2, 9.6, 23.8, 10, 12, 12.7, 12, 5.7, 8, 2.6, 0, 0,
         4.5, 0, 12.3, 13.5, 13.5, 6.7, 8, 11.7, 4.7, 10.5, 12.3, 7.2, 9.2, 0.9, 2.5, 2, 7.7, 0.3, 0.2, 0.4, 0, 0,
         0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.6, 0, 2.4, 6.1, 0.8, 1.1, 11.3, 1.9, 0.7, 0, 1, 1.3,
         0.5,
         15.6, 5.2, 3.9, 7.9, 10.1, 0, 10.3, 1.9, 16.6, 5.5, 11.2, 18.3, 8.4, 15.3, 27.8, 16.7, 14.3, 22.2, 12, 5.7,
         23.8, 5.8, 14.9, 18.5, 2.3, 8.1, 19.3, 14.5, 20.1, 19.2, 32.2, 26.2, 31.6, 9.8, 55.9, 35.5, 47.2, 31.5,
         33.5,
         37.2, 65, 26.3, 68.8, 73.7, 58.8, 44.3, 43.6, 38.8, 23.2, 47.8, 56.4, 38.1, 29.9, 36.4, 57.9, 96.2, 26.4,
         21.2,
         40, 50, 45, 36.7, 25.6, 28.9, 28.4, 34.9, 22.4, 25.4, 34.5, 53.1, 36.4, 28, 31.5, 26.1, 31.7, 10.9, 25.8,
         32.5,
         20.7, 3.7, 20.2, 19.6, 35, 31.4, 26.1, 14.9, 27.5, 25.1, 30.6, 19.2, 26.6, 4.5, 19.4, 29.3, 10.8, 20.6,
         25.9,
         5.2, 9, 7.9, 9.7, 21.5, 4.3, 5.7, 9.2, 1.7, 1.8, 2.5, 4.8, 4.4, 18.8, 4.4, 0, 0, 0.9, 16.1, 13.5, 1.5, 5.6,
         7.9, 2.1, 0, 0.4, 0, 0, 0, 0, 0.6, 0, 0, 0, 0.5, 0, 0, 0, 0, 20.4, 21.6, 10.8, 0, 19.4, 2.8, 0, 0, 1.4,
         20.5,
         25.2, 0, 0.8, 5, 15.5, 22.4, 3.8, 15.4, 15.4, 30.9, 25.4, 15.7, 15.6, 11.7, 22, 17.7, 18.2, 36.7, 24, 32.4,
         37.1, 52.5, 39.6, 18.9, 50.6, 39.5, 68.1, 34.6, 47.4, 57.8, 46, 56.3, 56.7, 42.9, 53.7, 49.6, 57.2, 48.2,
         46.1,
         52.8, 64.4, 65, 61.1, 89.1, 98, 54.3, 76.4, 50.4, 54.7, 57, 46.6, 43, 49.4, 72.3, 95, 67.5, 73.9, 90.8,
         78.3,
         52.8, 57.2, 67.6, 56.5, 52.2, 72.1, 84.6, 107.1, 66.3, 65.1, 43.9, 50.7, 62.1, 84.4, 81.2, 82.1, 47.5,
         50.1,
         93.4, 54.6, 38.1, 33.4, 45.2, 54.9, 37.9, 46.2, 43.5, 28.9, 30.9, 55.5, 55.1, 26.9, 41.3, 26.7, 13.9, 8.9,
         8.2,
         21.1, 14.3, 27.5, 11.3, 14.9, 11.8, 2.8, 12.9, 1, 7, 5.7, 11.6, 7.5, 5.9, 9.9, 4.9, 18.1, 3.9, 1.4, 8.8,
         7.8,
         8.7, 4, 11.5, 24.8, 30.5, 34.5, 7.5, 24.5, 19.7, 61.5, 43.6, 33.2, 59.8, 59, 100.8, 95.2, 100, 77.5, 88.6,
         107.6, 98.1, 142.9, 111.4, 124.7, 116.7, 107.8, 95.1, 137.4, 120.9, 206.2, 188, 175.6, 134.6, 138.2, 111.3,
         158, 162.8, 134, 96.3, 123.7, 107, 129.8, 144.9, 84.8, 140.8, 126.6, 137.6, 94.5, 108.2, 78.8, 73.6, 90.8,
         77.4, 79.8, 107.6, 102.5, 77.7, 61.8, 53.8, 54.6, 84.7, 131.2, 132.7, 90.8, 68.8, 63.6, 81.2, 87.7, 55.5,
         65.9,
         69.2, 48.5, 60.7, 57.8, 74, 49.8, 54.3, 53.7, 24, 29.9, 29.7, 42.6, 67.4, 55.7, 30.8, 39.3, 35.1, 28.5,
         19.8,
         38.8, 20.4, 22.1, 21.7, 26.9, 24.9, 20.5, 12.6, 26.5, 18.5, 38.1, 40.5, 17.6, 13.3, 3.5, 8.3, 8.8, 21.1,
         10.5,
         9.5, 11.8, 4.2, 5.3, 19.1, 12.7, 9.4, 14.7, 13.6, 20.8, 12, 3.7, 21.2, 23.9, 6.9, 21.5, 10.7, 21.6, 25.7,
         43.6,
         43.3, 56.9, 47.8, 31.1, 30.6, 32.3, 29.6, 40.7, 39.4, 59.7, 38.7, 51, 63.9, 69.2, 59.9, 65.1, 46.5, 54.8,
         107.1, 55.9, 60.4, 65.5, 62.6, 44.9, 85.7, 44.7, 75.4, 85.3, 52.2, 140.6, 161.2, 180.4, 138.9, 109.6,
         159.1,
         111.8, 108.9, 107.1, 102.2, 123.8, 139.2, 132.5, 100.3, 132.4, 114.6, 159.9, 156.7, 131.7, 96.5, 102.5,
         80.6,
         81.2, 78, 61.3, 93.7, 71.5, 99.7, 97, 78, 89.4, 82.6, 44.1, 61.6, 70, 39.1, 61.6, 86.2, 71, 54.8, 60, 75.5,
         105.4, 64.6, 56.5, 62.6, 63.2, 36.1, 57.4, 67.9, 62.5, 50.9, 71.4, 68.4, 67.5, 61.2, 65.4, 54.9, 46.9, 42,
         39.7, 37.5, 67.3, 54.3, 45.4, 41.1, 42.9, 37.7, 47.6, 34.7, 40, 45.9, 50.4, 33.5, 42.3, 28.8, 23.4, 15.4,
         20,
         20.7, 26.4, 24, 21.1, 18.7, 15.8, 22.4, 12.7, 28.2, 21.4, 12.3, 11.4, 17.4, 4.4, 9.1, 5.3, 0.4, 3.1, 0,
         9.7,
         4.3, 3.1, 0.5, 4.9, 0.4, 6.5, 0, 5, 4.6, 5.9, 4.4, 4.5, 7.7, 7.2, 13.7, 7.4, 5.2, 11.1, 29.2, 16, 22.2,
         16.9,
         42.4, 40.6, 31.4, 37.2, 39, 34.9, 57.5, 38.3, 41.4, 44.5, 56.7, 55.3, 80.1, 91.2, 51.9, 66.9, 83.7, 87.6,
         90.3,
         85.7, 91, 87.1, 95.2, 106.8, 105.8, 114.6, 97.2, 81, 81.5, 88, 98.9, 71.4, 107.1, 108.6, 116.7, 100.3,
         92.2,
         90.1, 97.9, 95.6, 62.3, 77.8, 101, 98.5, 56.8, 87.8, 78, 82.5, 79.9, 67.2, 53.7, 80.5, 63.1, 64.5, 43.6,
         53.7,
         64.4, 84, 73.4, 62.5, 66.6, 42, 50.6, 40.9, 48.3, 56.7, 66.4, 40.6, 53.8, 40.8, 32.7, 48.1, 22, 39.9, 37.7,
         41.2, 57.7, 47.1, 66.3, 35.8, 40.6, 57.8, 54.7, 54.8, 28.5, 33.9, 57.6, 28.6, 48.7, 39.3, 39.5, 29.4, 34.5,
         33.6, 26.8, 37.8, 21.6, 17.1, 24.6, 12.8, 31.6, 38.4, 24.6, 17.6, 12.9, 16.5, 9.3, 12.7, 7.3, 14.1, 9, 1.5,
         0,
         0.7, 9.2, 5.1, 2.9, 1.5, 5, 4.9, 9.8, 13.5, 9.3, 25.2, 15.6, 15.8, 26.5, 36.6, 26.7, 31.1, 28.6, 34.4,
         43.8,
         61.7, 59.1, 67.6, 60.9, 59.3, 52.7, 41, 104, 108.4, 59.2, 79.6, 80.6, 59.4, 77.4, 104.3, 77.3, 114.9,
         159.4,
         160, 176, 135.6, 132.4, 153.8, 136, 146.4, 147.5, 130, 88.3, 125.3, 143.2, 162.4, 145.5, 91.7, 103, 110,
         80.3,
         89, 105.4, 90.3, 79.5, 120.1, 88.4, 102.1, 107.6, 109.9, 105.5, 92.9, 114.6, 103.5, 112, 83.9, 86.7, 107,
         98.3,
         76.2, 47.9, 44.8, 66.9, 68.2, 47.5, 47.4, 55.4, 49.2, 60.8, 64.2, 46.4, 32, 44.6, 38.2, 67.8, 61.3, 28,
         34.3,
         28.9, 29.3, 14.6, 22.2, 33.8, 29.1, 11.5, 23.9, 12.5, 14.6, 2.4, 12.7, 17.7, 9.9, 14.3, 15, 31.2, 2.3, 5.1,
         1.6, 15.2, 8.8, 9.9, 14.3, 9.9, 8.2, 24.4, 8.7, 11.7, 15.8, 21.2, 13.4, 5.9, 6.3, 16.4, 6.7, 14.5, 2.3,
         3.3, 6,
         7.8, 0.1, 5.8, 6.4, 0.1, 0, 5.3, 1.1, 4.1, 0.5, 0.8, 0.6, 0, 6.2, 2.4, 4.8, 7.5, 10.7, 6.1, 12.3, 12.9,
         7.2,
         24, 27.5, 19.5, 19.3, 23.5, 34.1, 21.9, 48.1, 66, 43, 30.7, 29.6, 36.4, 53.2, 51.5, 51.7, 43.5, 60.5, 76.9,
         58,
         53.2, 64, 54.8, 47.3, 45, 69.3, 67.5, 95.8, 64.1, 45.2, 45.4, 40.4, 57.7, 59.2, 84.4, 41.8, 60.6, 46.9,
         42.8,
         82.1, 32.1, 76.5, 80.6, 46, 52.6, 83.8, 84.5, 75.9, 91.5, 86.9, 86.8, 76.1, 66.5, 51.2, 53.1, 55.8, 61.9,
         47.8,
         36.6, 47.2, 42.8, 71.8, 49.8, 55, 73, 83.7, 66.5, 50, 39.6, 38.7, 33.3, 21.7, 29.9, 25.9, 57.3, 43.7, 30.7,
         27.1, 30.3, 16.9, 21.4, 8.6, 0.3, 12.4, 10.3, 13.2, 4.2, 6.9, 20, 15.7, 23.3, 21.4, 7.4, 6.6, 6.9, 20.7,
         12.7,
         7.1, 7.8, 5.1, 7, 7.1, 3.1, 2.8, 8.8, 2.1, 10.7, 6.7, 0.8, 8.5, 7, 4.3, 2.4, 6.4, 9.7, 20.6, 6.5, 2.1, 0.2,
         6.7, 5.3, 0.6, 5.1, 1.6, 4.8, 1.3, 11.6, 8.5, 17.2, 11.2, 9.6, 7.8, 13.5, 22.2, 10.4, 20.5, 41.1, 48.3,
         58.8,
         33.2, 53.8, 51.5, 41.9, 32.3, 69.1, 75.6, 49.9, 69.6, 79.6, 76.3, 76.8, 101.4, 62.8, 70.5, 65.4, 78.6, 75,
         73,
         65.7, 88.1, 84.7, 88.2, 88.8, 129.2, 77.9, 79.7, 75.1, 93.8, 83.2, 84.6, 52.3, 81.6, 101.2, 98.9, 106,
         70.3,
         65.9, 75.5, 56.6, 60, 63.3, 67.2, 61, 76.9, 67.5, 71.5, 47.8, 68.9, 57.7, 67.9, 47.2, 70.7, 29, 57.4, 52,
         43.8,
         27.7, 49, 45, 27.2, 61.3, 28.4, 38, 42.6, 40.6, 29.4, 29.1, 31, 20, 11.3, 27.6, 21.8, 48.1, 14.3, 8.4,
         33.3,
         30.2, 36.4, 38.3, 14.5, 25.8, 22.3, 9, 31.4, 34.8, 34.4, 30.9, 12.6, 19.5, 9.2, 18.1, 14.2, 7.7, 20.5,
         13.5,
         2.9, 8.4, 13, 7.8, 10.5, 9.4, 13.6, 8.6, 16, 15.2, 12.1, 8.3, 4.3, 8.3, 12.9, 4.5, 0.3, 0.2, 2.4, 4.5, 0,
         10.2,
         5.8, 0.7, 1, 0.6, 3.7, 3.8, 0, 5.2, 0, 12.4, 0, 2.8, 1.4, 0.9, 2.3, 7.6, 16.3, 10.3, 1.1, 8.3, 17, 13.5,
         26.1,
         14.6, 16.3, 27.9, 28.8, 11.1, 38.9, 44.5, 45.6, 31.6, 24.5, 37.2, 43, 39.5, 41.9, 50.6, 58.2, 30.1, 54.2,
         38,
         54.6, 54.8, 85.8, 56.5, 39.3, 48, 49, 73, 58.8, 55, 78.7, 107.2, 55.5, 45.5, 31.3, 64.5, 55.3, 57.7, 63.2,
         103.6, 47.7, 56.1, 17.8, 38.9, 64.7, 76.4, 108.2, 60.7, 52.6, 42.9, 40.4, 49.7, 54.3, 85, 65.4, 61.5, 47.3,
         39.2, 33.9, 28.7, 57.6, 40.8, 48.1, 39.5, 90.5, 86.9, 32.3, 45.5, 39.5, 56.7, 46.6, 66.3, 32.3, 36, 22.6,
         35.8,
         23.1, 38.8, 58.4, 55.8, 54.2, 26.4, 31.5, 21.4, 8.4, 22.2, 12.3, 14.1, 11.5, 26.2, 38.3, 4.9, 5.8, 3.4, 9,
         7.8,
         16.5, 9, 2.2, 3.5, 4, 4, 2.6, 4.2, 2.2, 0.3, 0, 4.9, 4.5, 4.4, 4.1, 3, 0.3, 9.5, 4.6, 1.1, 6.4, 2.3, 2.9,
         0.5,
         0.9, 0, 0, 1.7, 0.2, 1.2, 3.1, 0.7, 3.8, 2.8, 2.6, 3.1, 17.3, 5.2, 11.4, 5.4, 7.7, 12.7, 8.2, 16.4, 22.3,
         23,
         42.3, 38.8, 41.3, 33, 68.8, 71.6, 69.6, 49.5, 53.5, 42.5, 34.5, 45.3, 55.4, 67, 71.8, 74.5, 67.7, 53.5,
         35.2,
         45.1, 50.7, 65.6, 53, 74.7, 71.9, 94.8, 74.7, 114.1, 114.9, 119.8, 154.5, 129.4, 72.2, 96.4, 129.3, 96,
         65.3,
         72.2, 80.5, 76.7, 59.4, 107.6, 101.7, 79.9, 85, 83.4, 59.2, 48.1, 79.5, 66.5, 51.8, 88.1, 111.2, 64.7, 69,
         54.7, 52.8, 42, 34.9, 51.1, 53.9, 70.2, 14.8, 33.3, 38.7, 27.5, 19.2, 36.3, 49.6, 27.2, 29.9, 31.5, 28.3,
         26.7,
         32.4, 22.2, 33.7, 41.9, 22.8, 17.8, 18.2, 17.8, 20.3, 11.8, 26.4, 54.7, 11, 8, 5.8, 10.9, 6.5, 4.7, 6.2,
         7.4,
         17.5, 4.5, 1.5, 3.3, 6.1, 3.2, 9.1, 3.5, 0.5, 13.2, 11.6, 10, 2.8, 0.5, 5.1, 1.8, 11.3, 20.8, 24, 28.1,
         19.3,
         25.1, 25.6, 22.5, 16.5, 5.5, 23.2, 18, 31.7, 42.8, 47.5, 38.5, 37.9, 60.2, 69.2, 58.6, 98.6, 71.8, 70,
         62.5,
         38.5, 64.3, 73.5, 52.3, 61.6, 60.8, 71.5, 60.5, 79.4, 81.6, 93, 69.6, 93.5, 79.1, 59.1, 54.9, 53.8, 68.4,
         63.1,
         67.2, 45.2, 83.5, 73.5, 85.4, 80.6, 76.9, 91.4, 98, 83.8, 89.7, 61.4, 50.3, 59, 68.9, 64.1, 50.2, 52.8,
         58.2,
         71.9, 70.2, 65.8, 34.4, 54, 81.1, 108, 65.3, 49.2, 35, 38.2, 36.8, 28.8, 21.9, 24.9, 32.1, 34.4, 35.6,
         25.8,
         14.6, 43.1, 30, 31.2, 24.6, 15.3, 17.4, 13, 19, 10, 18.7, 17.8, 12.1, 10.6, 11.2, 11.2, 17.9, 22.2, 9.6,
         6.8,
         4, 8.9, 8.2, 11, 12.3, 22.2, 10.1, 2.9, 3.2, 5.2, 2.8, 0.2, 5.1, 3, 0.6, 0.3, 3.4, 7.8, 4.3, 11.3, 19.7,
         6.7,
         9.3, 8.3, 4, 5.7, 8.7, 15.4, 18.9, 20.5, 23.1, 12.2, 27.3, 45.7, 33.9, 30.1, 42.1, 53.2, 64.2, 61.5, 62.8,
         74.3, 77.1, 74.9, 54.6, 70, 52.3, 87, 76, 89, 115.4, 123.4, 132.5, 128.5, 83.9, 109.3, 116.7, 130.3, 145.1,
         137.7, 100.7, 124.9, 74.4, 88.8, 98.4, 119.2, 86.5, 101, 127.4, 97.5, 165.3, 115.7, 89.6, 99.1, 122.2,
         92.7,
         80.3, 77.4, 64.6, 109.1, 118.3, 101, 97.6, 105.8, 112.6, 88.1, 68.1, 42.1, 50.5, 59.4, 83.3, 60.7, 54.4,
         83.9,
         67.5, 105.5, 66.5, 55, 58.4, 68.3, 45.6, 44.5, 46.4, 32.8, 29.5, 59.8, 66.9, 60, 65.9, 46.3, 38.3, 33.7,
         35.6,
         52.8, 54.2, 60.7, 25, 11.4, 17.7, 20.2, 17.2, 19.2, 30.7, 22.5, 12.4, 28.9, 27.4, 26.1, 14.1, 7.6, 13.2,
         19.4,
         10, 7.8, 10.2, 18.8, 3.7, 0.5, 11, 0.3, 2.5, 5, 5, 16.7, 14.3, 16.9, 10.8, 28.4, 18.5, 12.7, 21.5, 32,
         30.6,
         36.2, 42.6, 25.9, 34.9, 68.8, 46, 27.4, 47.6, 86.2, 76.6, 75.7, 84.9, 73.5, 116.2, 107.2, 94.4, 102.3,
         123.8,
         121.7, 115.7, 113.4, 129.8, 149.8, 201.3, 163.9, 157.9, 188.8, 169.4, 163.6, 128, 116.5, 108.5, 86.1, 94.8,
         189.7, 174, 167.8, 142.2, 157.9, 143.3, 136.3, 95.8, 138, 119.1, 182.3, 157.5, 147, 106.2, 121.7, 125.8,
         123.8,
         145.3, 131.6, 143.5, 117.6, 101.6, 94.8, 109.7, 113.4, 106.2, 83.6, 91, 85.2, 51.3, 61.4, 54.8, 54.1, 59.9,
         59.9, 59.9, 92.9, 108.5, 100.6, 61.5, 61, 83.1, 51.6, 52.4, 45.8, 40.7, 22.7, 22, 29.1, 23.4, 36.4, 39.3,
         54.9,
         28.2, 23.8, 22.1, 34.3, 26.5, 3.9, 10, 27.8, 12.5, 21.8, 8.6, 23.5, 19.3, 8.2, 1.6, 2.5, 0.2, 0.5, 10.9,
         1.8,
         0.8, 0.2, 4.8, 8.4, 1.5, 7, 9.2, 7.6, 23.1, 20.8, 4.9, 11.3, 28.9, 31.7, 26.7, 40.7, 42.7, 58.5, 89.2,
         76.9,
         73.6, 124, 118.4, 110.7, 136.6, 116.6, 129.1, 169.6, 173.2, 155.3, 201.3, 192.1, 165, 130.2, 157.4, 175.2,
         164.6, 200.7, 187.2, 158, 235.8, 253.8, 210.9, 239.4, 202.5, 164.9, 190.7, 196, 175.3, 171.5, 191.4, 200.2,
         201.2, 181.5, 152.3, 187.6, 217.4, 143.1, 185.7, 163.3, 172, 168.7, 149.6, 199.6, 145.2, 111.4, 124, 125,
         146.3, 106, 102.2, 122, 119.6, 110.2, 121.7, 134.1, 127.2, 82.8, 89.6, 85.6, 57.9, 46.1, 53, 61.4, 51,
         77.4,
         70.2, 55.9, 63.6, 37.7, 32.6, 40, 38.7, 50.3, 45.6, 46.4, 43.7, 42, 21.8, 21.8, 51.3, 39.5, 26.9, 23.2,
         19.8,
         24.4, 17.1, 29.3, 43, 35.9, 19.6, 33.2, 38.8, 35.3, 23.4, 14.9, 15.3, 17.7, 16.5, 8.6, 9.5, 9.1, 3.1, 9.3,
         4.7,
         6.1, 7.4, 15.1, 17.5, 14.2, 11.7, 6.8, 24.1, 15.9, 11.9, 8.9, 16.8, 20.1, 15.8, 17, 28.2, 24.4, 25.3, 48.7,
         45.3, 47.7, 56.7, 51.2, 50.2, 57.2, 57.2, 70.4, 110.9, 93.6, 111.8, 69.5, 86.5, 67.3, 91.5, 107.2, 76.8,
         88.2,
         94.3, 126.4, 121.8, 111.9, 92.2, 81.2, 127.2, 110.3, 96.1, 109.3, 117.2, 107.7, 86, 109.8, 104.4, 120.5,
         135.8,
         106.8, 120, 106, 96.8, 98, 91.3, 95.7, 93.5, 97.9, 111.5, 127.8, 102.9, 109.5, 127.5, 106.8, 112.5, 93,
         99.5,
         86.6, 95.2, 83.5, 91.3, 79, 60.7, 71.8, 57.5, 49.8, 81, 61.4, 50.2, 51.7, 63.2, 82.2, 61.5, 88.4, 80.1,
         63.2,
         80.5, 88, 76.5, 76.8, 64, 61.3, 41.6, 45.3, 43.4, 42.9, 46, 57.7, 42.4, 39.5, 23.1, 25.6, 59.3, 30.7, 23.9,
         23.3, 27.6, 26, 21.3, 40.3, 39.5, 36, 55.8, 33.6, 40.2, 47.1, 25, 20.5, 18.9, 11.5, 11.5, 5.1, 9, 11.4,
         28.2,
         39.7, 13.9, 9.1, 19.4, 7.8, 8.1, 4.3, 21.9, 18.8, 12.4, 12.2, 1.9, 16.4, 13.5, 20.6, 5.2, 15.3, 16.4, 23.1,
         8.7, 12.9, 18.6, 38.5, 21.4, 30.1, 44, 43.8, 29.1, 43.2, 51.9, 93.6, 76.5, 99.7, 82.7, 95.1, 70.4, 58.1,
         138.2,
         125.1, 97.9, 122.7, 166.6, 137.5, 138, 101.5, 134.4, 149.5, 159.4, 142.2, 188.4, 186.2, 183.3, 176.3,
         159.6,
         155, 126.2, 164.1, 179.9, 157.3, 136.3, 135.4, 155, 164.7, 147.9, 174.4, 114, 141.3, 135.5, 156.4, 127.5,
         90,
         143.8, 158.7, 167.3, 162.4, 137.5, 150.1, 111.2, 163.6, 153.8, 122, 82.2, 110.4, 106.1, 107.6, 118.8, 94.7,
         98.1, 127, 84.3, 51, 66.5, 80.7, 99.2, 91.1, 82.2, 71.8, 50.3, 55.8, 33.3, 33.4])
    return ts


def nile():
    """
    Description
    -----------
    Measurements of the annual flow of the river Nile at Aswan (formerly Assuan), 1871–1970, in 10^8 m^3, “with apparent changepoint near 1898” (Cobb(1978), Table 1, p.249).

    Format
    ------
    A time series of length 100.

    Source
    ------
    Durbin, J. and Koopman, S. J. (2001). Time Series Analysis by State Space Methods. Oxford University Press. http://www.ssfpack.com/DKbook.html

    References
    ----------
    [1] Balke, N. S. (1993). Detecting level shifts in time series. Journal of Business and Economic Statistics, 11, 81–92. doi: 10.2307/1391308.

    [2] Cobb, G. W. (1978). The problem of the Nile: conditional solution to a change-point problem. Biometrika 65, 243–51. doi: 10.2307/2335202.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting sunspots data.
    """
    ts = Series(
        [1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935, 1110, 994, 1020, 960, 1180, 799, 958,
         1140,
         1100, 1210, 1150, 1250, 1260, 1220, 1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916, 692, 1020, 1050,
         969,
         831, 726, 456, 824, 702, 1120, 1100, 832, 764, 821, 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759, 781,
         865, 845, 944, 984, 897, 822, 1010, 771, 676, 649, 846, 812, 742, 801, 1040, 860, 874, 848, 890, 744, 749,
         838,
         1050, 918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746, 919, 718, 714, 740])
    return ts


def wineind():
    """

    Description
    -----------
    Australian total wine sales by wine makers in bottles <= 1 litre. This time-series records wine sales by Australian wine makers between Jan 1980 -- Aug 1994. This dataset is found in the R ``forecast`` package. This is monthly data.

    References
    ----------
    [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/wineind  # noqa: E501

    Returns
    -------
    ts : Series
        Returns a pandas series consisting wineind data.

    """
    ts = Series(
        [15136, 16733, 20016, 17708, 18019, 19227, 22893, 23739, 21133, 22591, 26786, 29740, 15028, 17977, 20008,
         21354,
         19498, 22125, 25817, 28779, 20960, 22254, 27392, 29945, 16933, 17892, 20533, 23569, 22417, 22084, 26580,
         27454,
         24081, 23451, 28991, 31386, 16896, 20045, 23471, 21747, 25621, 23859, 25500, 30998, 24475, 23145, 29701,
         34365,
         17556, 22077, 25702, 22214, 26886, 23191, 27831, 35406, 23195, 25110, 30009, 36242, 18450, 21845, 26488,
         22394,
         28057, 25451, 24872, 33424, 24052, 28449, 33533, 37351, 19969, 21701, 26249, 24493, 24603, 26485, 30723,
         34569,
         26689, 26157, 32064, 38870, 21337, 19419, 23166, 28286, 24570, 24001, 33151, 24878, 26804, 28967, 33311,
         40226,
         20504, 23060, 23562, 27562, 23940, 24584, 34303, 25517, 23494, 29095, 32903, 34379, 16991, 21109, 23740,
         25552,
         21752, 20294, 29009, 25500, 24166, 26960, 31222, 38641, 14672, 17543, 25453, 32683, 22449, 22316, 27595,
         25451,
         25421, 25288, 32568, 35110, 16052, 22146, 21198, 19543, 22084, 23816, 29961, 26773, 26635, 26972, 30207,
         38687,
         16974, 21697, 24179, 23757, 25013, 24019, 30345, 24488, 25156, 25650, 30923, 37240, 17466, 19463, 24352,
         26805,
         25236, 24735, 29356, 31234, 22724, 28496, 32857, 37198, 13652, 22784, 23565, 26323, 23779, 27549, 29660,
         23356])
    return ts


def co2 ():
    """
    Description
    -----------
    Atmospheric concentrations of CO2 are expressed in parts per million (ppm) and reported in the preliminary 1997 SIO manometric mole fraction scale.

    Format
    ------
    A pandas.series containing 468 observations; monthly from 1959 to 1997.

    Details
    -------
    The values for February, March and April of 1964 were missing and have been obtained by interpolating linearly between the values for January and May of 1964.

    Source
    ------
    Keeling, C. D. and Whorf, T. P., Scripps Institution of Oceanography (SIO), University of California, La Jolla, California USA 92093-0220.
    ftp://cdiac.esd.ornl.gov/pub/maunaloa-co2/maunaloa.co2.

    References
    ----------
    Cleveland, W. S. (1993) Visualizing Data. New Jersey: Summit Press.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting wineind data.
    """
    ts = Series(
        [315.42, 316.31, 316.5, 317.56, 318.13, 318.0, 316.39, 314.65, 313.68, 313.18, 314.66, 315.43, 316.27,
         316.81,
         317.42, 318.87, 319.87, 319.43, 318.01, 315.74, 314.0, 313.68, 314.84, 316.03, 316.73, 317.54, 318.38,
         319.31,
         320.42, 319.61, 318.42, 316.63, 314.83, 315.16, 315.94, 316.85, 317.78, 318.4, 319.53, 320.42, 320.85,
         320.45,
         319.45, 317.25, 316.11, 315.27, 316.53, 317.53, 318.58, 318.92, 319.7, 321.22, 322.08, 321.31, 319.58,
         317.61,
         316.05, 315.83, 316.91, 318.2, 319.41, 320.07, 320.74, 321.4, 322.06, 321.73, 320.27, 318.54, 316.54,
         316.71,
         317.53, 318.55, 319.27, 320.28, 320.73, 321.97, 322.0, 321.71, 321.05, 318.71, 317.66, 317.14, 318.7,
         319.25,
         320.46, 321.43, 322.23, 323.54, 323.91, 323.59, 322.24, 320.2, 318.48, 317.94, 319.63, 320.87, 322.17,
         322.34,
         322.88, 324.25, 324.83, 323.93, 322.38, 320.76, 319.1, 319.24, 320.56, 321.8, 322.4, 322.99, 323.73,
         324.86,
         325.4, 325.2, 323.98, 321.95, 320.18, 320.09, 321.16, 322.74, 323.83, 324.26, 325.47, 326.5, 327.21,
         326.54,
         325.72, 323.5, 322.22, 321.62, 322.69, 323.95, 324.89, 325.82, 326.77, 327.97, 327.91, 327.5, 326.18,
         324.53,
         322.93, 322.9, 323.85, 324.96, 326.01, 326.51, 327.01, 327.62, 328.76, 328.4, 327.2, 325.27, 323.2, 323.4,
         324.63, 325.85, 326.6, 327.47, 327.58, 329.56, 329.9, 328.92, 327.88, 326.16, 324.68, 325.04, 326.34,
         327.39,
         328.37, 329.4, 330.14, 331.33, 332.31, 331.9, 330.7, 329.15, 327.35, 327.02, 327.99, 328.48, 329.18,
         330.55,
         331.32, 332.48, 332.92, 332.08, 331.01, 329.23, 327.27, 327.21, 328.29, 329.41, 330.23, 331.25, 331.87,
         333.14,
         333.8, 333.43, 331.73, 329.9, 328.4, 328.17, 329.32, 330.59, 331.58, 332.39, 333.33, 334.41, 334.71,
         334.17,
         332.89, 330.77, 329.14, 328.78, 330.14, 331.52, 332.75, 333.24, 334.53, 335.9, 336.57, 336.1, 334.76,
         332.59,
         331.42, 330.98, 332.24, 333.68, 334.8, 335.22, 336.47, 337.59, 337.84, 337.72, 336.37, 334.51, 332.6,
         332.38,
         333.75, 334.78, 336.05, 336.59, 337.79, 338.71, 339.3, 339.12, 337.56, 335.92, 333.75, 333.7, 335.12,
         336.56,
         337.84, 338.19, 339.91, 340.6, 341.29, 341.0, 339.39, 337.43, 335.72, 335.84, 336.93, 338.04, 339.06,
         340.3,
         341.21, 342.33, 342.74, 342.08, 340.32, 338.26, 336.52, 336.68, 338.19, 339.44, 340.57, 341.44, 342.53,
         343.39,
         343.96, 343.18, 341.88, 339.65, 337.81, 337.69, 339.09, 340.32, 341.2, 342.35, 342.93, 344.77, 345.58,
         345.14,
         343.81, 342.21, 339.69, 339.82, 340.98, 342.82, 343.52, 344.33, 345.11, 346.88, 347.25, 346.62, 345.22,
         343.11,
         340.9, 341.18, 342.8, 344.04, 344.79, 345.82, 347.25, 348.17, 348.74, 348.07, 346.38, 344.51, 342.92,
         342.62,
         344.06, 345.38, 346.11, 346.78, 347.68, 349.37, 350.03, 349.37, 347.76, 345.73, 344.68, 343.99, 345.48,
         346.72,
         347.84, 348.29, 349.23, 350.8, 351.66, 351.07, 349.33, 347.92, 346.27, 346.18, 347.64, 348.78, 350.25,
         351.54,
         352.05, 353.41, 354.04, 353.62, 352.22, 350.27, 348.55, 348.72, 349.91, 351.18, 352.6, 352.92, 353.53,
         355.26,
         355.52, 354.97, 353.75, 351.52, 349.64, 349.83, 351.14, 352.37, 353.5, 354.55, 355.23, 356.04, 357.0,
         356.07,
         354.67, 352.76, 350.82, 351.04, 352.69, 354.07, 354.59, 355.63, 357.03, 358.48, 359.22, 358.12, 356.06,
         353.92,
         352.05, 352.11, 353.64, 354.89, 355.88, 356.63, 357.72, 359.07, 359.58, 359.17, 356.94, 354.92, 352.94,
         353.23,
         354.09, 355.33, 356.63, 357.1, 358.32, 359.41, 360.23, 359.55, 357.53, 355.48, 353.67, 353.95, 355.3,
         356.78,
         358.34, 358.89, 359.95, 361.25, 361.67, 360.94, 359.55, 357.49, 355.84, 356.0, 357.59, 359.05, 359.98,
         361.03,
         361.66, 363.48, 363.82, 363.3, 361.94, 359.5, 358.11, 357.8, 359.61, 360.74, 362.09, 363.29, 364.06,
         364.76,
         365.45, 365.01, 363.7, 361.54, 359.51, 359.65, 360.8, 362.38, 363.23, 364.06, 364.61, 366.4, 366.84,
         365.68,
         364.52, 362.57, 360.24, 360.83, 362.49, 364.34])
    return ts


def gas ():
    """
    Description
    -----------
    Australian monthly gas production: 1956–1995.

    Format
    ------
    Pandas Series containing gas time series

    Source
    -------
    Australian Bureau of Statistics.

    Returns
    -------
    ts : Series
        Returns a pandas series consisting wineind data.
    """
    ts = Series(
        [1709, 1646, 1794, 1878, 2173, 2321, 2468, 2416, 2184, 2121, 1962, 1825, 1751, 1688, 1920, 1941, 2311, 2279,
         2638, 2448, 2279, 2163, 1941, 1878, 1773, 1688, 1783, 1984, 2290, 2511, 2712, 2522, 2342, 2195, 1931, 1910,
         1730, 1688, 1899, 1994, 2342, 2553, 2712, 2627, 2363, 2311, 2026, 1910, 1762, 1815, 2005, 2089, 2617, 2828,
         2965, 2891, 2532, 2363, 2216, 2026, 1804, 1773, 2015, 2089, 2627, 2712, 3007, 2880, 2490, 2237, 2205, 1984,
         1868, 1815, 2047, 2142, 2743, 2775, 3028, 2965, 2501, 2501, 2131, 2015, 1910, 1868, 2121, 2268, 2690, 2933,
         3218, 3028, 2659, 2406, 2258, 2057, 1889, 1984, 2110, 2311, 2785, 3039, 3229, 3070, 2659, 2543, 2237, 2142,
         1962, 1910, 2216, 2437, 2817, 3123, 3345, 3112, 2659, 2469, 2332, 2110, 1910, 1941, 2216, 2342, 2923, 3229,
         3513, 3355, 2849, 2680, 2395, 2205, 1994, 1952, 2290, 2395, 2965, 3239, 3608, 3524, 3018, 2648, 2363, 2247,
         1994, 1941, 2258, 2332, 3323, 3608, 3957, 3672, 3155, 2933, 2585, 2384, 2057, 2100, 2458, 2638, 3292, 3724,
         4652, 4379, 4231, 3756, 3429, 3461, 3345, 4220, 4874, 5064, 5951, 6774, 7997, 7523, 7438, 6879, 6489, 6288,
         5919, 6183, 6594, 6489, 8040, 9715, 9714, 9756, 8595, 7861, 7753, 8154, 7778, 7402, 8903, 9742, 11372,
         12741,
         13733, 13691, 12239, 12502, 11241, 10829, 11569, 10397, 12493, 11962, 13974, 14945, 16805, 16587, 14225,
         14157,
         13016, 12253, 11704, 12275, 13695, 14082, 16555, 17339, 17777, 17592, 16194, 15336, 14208, 13116, 12354,
         12682,
         14141, 14989, 16159, 18276, 19157, 18737, 17109, 17094, 15418, 14312, 13260, 14990, 15975, 16770, 19819,
         20983,
         22001, 22337, 20750, 19969, 17293, 16498, 15117, 16058, 18137, 18471, 21398, 23854, 26025, 25479, 22804,
         19619,
         19627, 18488, 17243, 18284, 20226, 20903, 23768, 26323, 28038, 26776, 22886, 22813, 22404, 19795, 18839,
         18892,
         20823, 22212, 25076, 26884, 30611, 30228, 26762, 25885, 23328, 21930, 21433, 22369, 24503, 25905, 30605,
         34984,
         37060, 34502, 31793, 29275, 28305, 25248, 27730, 27424, 32684, 31366, 37459, 41060, 43558, 42398, 33827,
         34962,
         33480, 32445, 30715, 30400, 31451, 31306, 40592, 44133, 47387, 41310, 37913, 34355, 34607, 28729, 26138,
         30745,
         35018, 34549, 40980, 42869, 45022, 40387, 38180, 38608, 35308, 30234, 28801, 33034, 35294, 33181, 40797,
         42355,
         46098, 42430, 41851, 39331, 37328, 34514, 32494, 33308, 36805, 34221, 41020, 44350, 46173, 44435, 40943,
         39269,
         35901, 32142, 31239, 32261, 34951, 38109, 43168, 45547, 49568, 45387, 41805, 41281, 36068, 34879, 32791,
         34206,
         39128, 40249, 43519, 46137, 56709, 52306, 49397, 45500, 39857, 37958, 35567, 37696, 42319, 39137, 47062,
         50610,
         54457, 54435, 48516, 43225, 42155, 39995, 37541, 37277, 41778, 41666, 49616, 57793, 61884, 62400, 50820,
         51116,
         45731, 42528, 40459, 40295, 44147, 42697, 52561, 56572, 56858, 58363, 45627, 45622, 41304, 36016, 35592,
         35677,
         39864, 41761, 50380, 49129, 55066, 55671, 49058, 44503, 42145, 38698, 38963, 38690, 39792, 42545, 50145,
         58164,
         59035, 59408, 55988, 47321, 42269, 39606, 37059, 37963, 31043, 41712, 50366, 56977, 56807, 54634, 51367,
         48073,
         46251, 43736, 39975, 40478, 46895, 46147, 55011, 57799, 62450, 63896, 57784, 53231, 50354, 38410, 41600,
         41471,
         46287, 49013, 56624, 61739, 66600, 60054])
    return ts