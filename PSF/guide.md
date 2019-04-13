# Introduction to Pattern Sequence based Forecasting (PSF) algorithm
Package: PSF

Introduction

The Algorithm Pattern Sequence based Forecasting (PSF) was first proposed by Martinez Alvarez, et al., 2008 and then modified and suggested improvement by Martinez Alvarez, et al., 2011. The technical detailes are mentioned in referenced articles. PSF algorithm consists of various statistical operations like:

* Data Normalization/ Denormalization
* Calculation of optimum Window size (W)
* Calculation of optimum cluster size (k)
* Pattern Sequence based Forecasting
* RMSE/MAE Calculation, etc..


The following code illustrates PSF package:
```python   
from PSF import Psf
import pandas as pd

# Importing csv file
data = pd.read_csv('nottem.csv')

# Extracting Time Series
ts = data['nottem']

# Creating PSF model for prediction.
a = Psf(data = ts, cycle = 12)

# Use predict to predict the values
a.predict(n_ahead = 12)

# Print the model
a.model_print()

# Plot the time series
a.psf_plot(ts, a.preds)
```    
Output : 
    
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
    Name: nottem, Length: 240, dtype: float64
    
    Predicted Values : 
     [39.4 40.9 42.4 47.8 52.4 58.  60.7 61.8 58.2 46.7 46.6 37.8]
    
    k =  3
    
    w =  12
    
    cycle =  12
    

![](.README_images/cbdd3852.png)

Example:
```python
from PSF import Psf
import pandas as pd

# Importing csv file
data = pd.read_csv('penguin.csv')

# Extracting Time Series
ts = data[data.columns.values[1]]

# Creating PSF model for prediction.
a = Psf(data=ts, cycle=12)

# Use predict to predict the values
a.predict(n_ahead=12)

# Print the model
a.model_print()

# Plot the time series
a.psf_plot(ts, a.preds)
```

Output:

    Original time-series : 
    0      753
    1      448
    2      356
    3      504
    4      698
    5      256
    6      361
    7      476
    8      541
    9      812
    10     914
    11     998
    12     762
    13     461
    14     374
    15     521
    16     712
    17     274
    18     384
    19     492
    20     561
    21     821
    22     930
    23    1014
    24     779
    25     478
    26     391
    27     543
    28     910
    29     287
          ... 
    54     225
    55     304
    56     416
    57     642
    58     769
    59     853
    60     572
    61     273
    62     208
    63     341
    64     553
    65     136
    66     231
    67     299
    68     403
    69     632
    70     759
    71     848
    72     561
    73     268
    74     212
    75     331
    76     542
    77     128
    78     225
    79     301
    80     389
    81     624
    82     748
    83     842
    Name: Number, Length: 84, dtype: int64
    
    Predicted Values : 
     [572. 273. 208. 341. 553. 136. 231. 299. 403. 632. 759. 848.]
    
    k =  3
    
    w =  14
    
    cycle =  12
![](.guide_images/acb2487e.png)