import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import copy
from itertools import cycle
from decomposition_functions import *


year = 2019

training_start_1 = '2019-01-01 00:00:00+01:00'
training_end_1   = '2019-01-21 23:00:00+01:00'
test_start_1     = '2019-01-29 00:00:00+01:00'
test_end_1       = '2019-02-04 23:00:00+01:00'

training_start_2 = '2019-07-09 00:00:00+01:00'
training_end_2   = '2019-07-29 23:00:00+01:00'
test_start_2     = '2019-07-24 00:00:00+01:00'
test_end_2       = '2019-07-26 23:00:00+01:00'

df1 = pd.read_csv('https://data.stadt-zuerich.ch/dataset/ugz_meteodaten_stundenmittelwerte/'
                          'download/ugz_ogd_meteo_h1_' + str(year) + '.csv')


# convert dates from string to datetime objects
df1['Datum'] = pd.to_datetime(df1['Datum'])


# filter according to one location
filtr = (df1['Standort'] == 'Zch_Rosengartenstrasse')
df1_t = df1.loc[filtr]
df1_t.reset_index(drop=True, inplace=True)


# filter according to one location
filtr = (df1['Standort'] == 'Zch_Stampfenbachstrasse')
df1 = df1.loc[filtr]
df1.reset_index(drop=True, inplace=True)

# create a new DataFrame with one column for each variable
df2_t = pd.DataFrame()
df2_t['timestamp'] = pd.unique(df1_t['Datum'])
df2_t['temperature'] = df1_t.loc[df1_t['Parameter'] == 'T']['Wert'].reset_index(drop=True)
df2_t['humidity'] = df1_t.loc[df1_t['Parameter'] == 'Hr']['Wert'].reset_index(drop=True)


# create a new DataFrame with one column for each variable
df2 = pd.DataFrame()
df2['timestamp'] = pd.unique(df1['Datum'])
df2['temperature'] = df1.loc[df1['Parameter'] == 'T']['Wert'].reset_index(drop=True)
df2['humidity'] = df1.loc[df1['Parameter'] == 'Hr']['Wert'].reset_index(drop=True)

del df1

# fill missing values
for i in df2.index:
    if np.isnan(df2.loc[i, 'temperature']):
        df2.loc[i, 'temperature'] = df2_t.loc[i, 'temperature']
    if np.isnan(df2.loc[i, 'humidity']):
        df2.loc[i, 'humidity'] = df2_t.loc[i, 'humidity']


# training and test data first interval 
filtr = (df2['timestamp'] >= training_start_1) & (df2['timestamp'] <= training_end_1)
training_data_1 = df2.loc[filtr]
training_data_1.reset_index(drop=True, inplace=True)

filtr = (df2['timestamp'] >= test_start_1) & (df2['timestamp'] <= test_end_1)
test_data_1 = df2.loc[filtr]
test_data_1.reset_index(drop=True, inplace=True)

# training and test data second interval 
filtr = (df2['timestamp'] >= training_start_2) & (df2['timestamp'] <= training_end_2)
training_data_2 = df2.loc[filtr]
training_data_2.reset_index(drop=True, inplace=True)

filtr = (df2['timestamp'] >= test_start_2) & (df2['timestamp'] <= test_end_2)
test_data_2 = df2.loc[filtr]
test_data_2.reset_index(drop=True, inplace=True)


# call the decompose function
y = decompose(training_data_2.loc[:, ['timestamp', 'humidity']], 24, 'additive')

# plot the decomposition
plot(y, '', 'time', 'humidity [%Hr]', 'additive')

