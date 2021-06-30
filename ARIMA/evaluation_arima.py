# imports

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import copy
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from arima_functions import *


# data sourcing and data preparation
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

# filling missing values

for i in df2.index:
    if np.isnan(df2.loc[i, 'temperature']):
        df2.loc[i, 'temperature'] = df2_t.loc[i, 'temperature']
    if np.isnan(df2.loc[i, 'humidity']):
        df2.loc[i, 'humidity'] = df2_t.loc[i, 'humidity']



# evaluation with 41 intervals

h = 24
variable = 'temperature' # or 'humidity'

MAE = []
MSE = []
RMSE = []


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

    sarima = pm.auto_arima(training_data[variable], error_action='ignore', trace=False,
                      suppress_warnings=True, maxiter=10,
                      seasonal=True, m=24)
    
    
    para = sarima.get_params()
    
    print(para['order'], para['seasonal_order'], para['with_intercept'], para['trend'])

    
    # evaluate the selected method on the test data
    res = rolling_forecasting_origin_evaluation(training_data, test_data, ARIMA, para, h)
        
    
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



print(evaluation)



