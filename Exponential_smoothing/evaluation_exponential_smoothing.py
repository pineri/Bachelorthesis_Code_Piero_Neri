# imports

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import copy
from exponential_smoothing_functions import *

# data sourcing and preparation

year = 2019

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



# evaluation strategy with 49 intervals

now = datetime.now()
print(now)

variable = 'temperature' # or 'humidity'
m = 24
h = 24
initial_grid = 6
fine_grid = 11
rand_iter = 500

MAE = []
MSE = []
RMSE = []

freq = {'N-N': 0, 'N-A': 0, 'N-M': 0, 'A-N': 0, 'A-A': 0, 'A-M': 0, 'Ad-N': 0, 'Ad-A': 0, 'Ad-M': 0}

# tuples of the form (methodName, function, hasSeasonality)
methods = [
           ('N-N', auto_N_N, False), ('N-A', auto_N_A, True), ('N-M', auto_N_M, True),
           ('A-N', auto_A_N, False), ('A-A', auto_A_A, True), ('A-M', auto_A_M, True),
           ('Ad-N', auto_Ad_N, False), ('Ad-A', auto_Ad_A, True), ('Ad-M', auto_Ad_M, True),
           ]

one_hour = df2.loc[1, 'timestamp'] - df2.loc[0, 'timestamp']
one_day = 24*one_hour
one_week = 7*one_day
three_weeks = 3 * one_week

# initialize the first interval of training and test data
start = df2.loc[0, 'timestamp']
split = start + three_weeks

filtr = (df2['timestamp'] >= start) & (df2['timestamp'] < split)
training_data = df2.loc[filtr, ['timestamp', variable]]
training_data.reset_index(drop=True, inplace=True)

filtr = (df2['timestamp'] >= split) & (df2['timestamp'] < split + one_week)
test_data = df2.loc[filtr, ['timestamp', variable]]
test_data.reset_index(drop=True, inplace=True)

cnt = 0
while(split+one_week <= df2.loc[df2.shape[0]-1, 'timestamp']):
    
    cnt += 1
    print(cnt)
    # find the method with the best AIC
    min_aic = 'start'
    best_method = None
    
    for method in methods:
        
        if(method[2]):
            res = method[1](training_data, m, h, initial_grid, fine_grid, rand_iter)
            aic = res[2]
            #print(aic)
            
        else:
            res = method[1](training_data, h, initial_grid, fine_grid, rand_iter)
            aic = res[2]
            #print(aic)
    
        if(min_aic == 'start' or aic < min_aic):
            min_aic = aic
            best_method = method
            
    
    freq[best_method[0]] += 1
    print(best_method[0])
    print(min_aic)
    
    # evaluate the selected method on the test data
    if(best_method[2]):
        res = rolling_forecasting_origin_evaluation(training_data, test_data, best_method[1], m, h, initial_grid, fine_grid, rand_iter)
        
    else:
        res = rolling_forecasting_origin_evaluation(training_data, test_data, best_method[1], 0, h, initial_grid, fine_grid, rand_iter)
    
    MAE.append( res[0] )
    MSE.append( res[1] )
    RMSE.append( res[2] )
    
    
    # shift training and test window by one week  
    start = start + one_week
    split = start + three_weeks

    filtr = (df2['timestamp'] >= start) & (df2['timestamp'] < split)
    training_data = df2.loc[filtr, ['timestamp', variable]]
    training_data.reset_index(drop=True, inplace=True)

    filtr = (df2['timestamp'] >= split) & (df2['timestamp'] < split + one_week)
    test_data = df2.loc[filtr, ['timestamp', variable]]
    test_data.reset_index(drop=True, inplace=True)
    

MAE_final = []
MSE_final = []
RMSE_final = []
    
MAE = np.array(MAE)
MSE = np.array(MSE)
RMSE = np.array(RMSE)

n_cols = MAE.shape[1]

for col in range(n_cols):
    MAE_final.append( np.mean(MAE[:, col]) )
    MSE_final.append( np.mean(MSE[:, col]) )
    RMSE_final.append( np.mean(RMSE[:, col]) )
    
evaluation = pd.DataFrame({'MAE': MAE_final,
                           'MSE': MSE_final,
                           'RMSE': RMSE_final})

now = datetime.now()
print(now)



print(evaluation)



print(freq)

