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



# uses the package pmdarima
def ARIMA(data, para, h):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    arimaa = pm.arima.ARIMA(order=para['order'], seasonal_order=para['seasonal_order'], maxiter=para['maxiter'],
             suppress_warnings=para['suppress_warnings'], trend=para['trend'], 
              with_intercept=para['with_intercept'])
    
    arimaa2 = arimaa.fit(y['y'])
    
    best_forecasts = list(arimaa2.predict(h))
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
    
    return forecasts



def rolling_forecasting_origin_evaluation(training_data, test_data, method, para , h=1):
    
    training_data.columns = ['t', 'y']
    test_data.columns = ['t', 'y']
    

    errors = []
    
    assert(h <= test_data.shape[0])

    for i in range(h-1, test_data.shape[0]): 

        result = method(training_data, para, h)
        
        
        errors.append(list(np.array(test_data['y'])[i-h+1:i+1] - np.array(result['y'])))
    
        training_data = pd.concat([training_data.loc[1:,], pd.DataFrame(test_data.loc[i-h+1:i-h+1,])])
        training_data.reset_index(drop=True, inplace=True)
        
    
    errors = np.array(errors)
    n_rows = errors.shape[0]
    n_cols = errors.shape[1]
    
    MAE = []
    MSE = []
    RMSE = []
    
    test_data = np.array(test_data['y'])
    
    for col in range(n_cols):
    
        MAE.append( ((1/n_rows)*np.abs(errors[:, col])).sum() )
        MSE.append( ((1/n_rows)*errors[:, col]**2).sum() )
        RMSE.append( ((1/n_rows)*errors[:, col]**2).sum()**0.5 )
    
    return (MAE, MSE, RMSE) 




