# imports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import copy
from itertools import cycle



def trend(y, m, dec_type):

    # check if the period m is even  
    if(m%2 == 0):
        mm = m + 1
    else:
        mm = m

    y.columns = ['t', 'y']

    # calculate the weights
    a = pd.Series(1/m for i in range(0, mm))

    if(m%2 == 0):
        a[0] /= 2
        a[m] /= 2
        
    start = int(m/2)
    end = y.shape[0] - start

    k = start

    trend_values = [None for v in range(y.shape[0])]

    # the for loop reflects the formula (6) explained above
    for t in range(start, end):
        T = y.loc[t-k:t+k, 'y']
        T.reset_index(drop=True, inplace=True)
        T = sum(T * a)
        trend_values[t] = T
        
    # if a trend value is zero the multiplicative dec_type cannot be used
    if(dec_type[0] == 'm'):
        if(0 in trend_values):
            raise ValueError('Multiplicative dec_type is not possible when a trend value is zero. '
                             'It would lead to division by zero.')

    y['T'] = trend_values

    return y



def seasonality(y, m, dec_type):

    # calculate the detrended values for the additive and the multiplicative dec_type correspondingly
    if(dec_type[0] == 'a'):
        deTrended = y.loc[:, 'y'] - y.loc[:, 'T']
    else:
        deTrended = y.loc[:, 'y'] / y.loc[:, 'T']

    # call the seasonalize function and calculate the mean for each seasonal component
    means = seasonalize(deTrended, m)

    # replicate the seasonal components for the following periods
    means_cycle = cycle(means)
    y['S'] = [next(means_cycle) for i in range(y.shape[0])]

    if(dec_type[0] == 'a'):
        # the seasonal components have to sum up to zero
        y['S'] -= sum(y['S']) / y.shape[0]
    else:
        # the seasonal components have to sum up to one
        y['S'] /= sum(y['S']) / y.shape[0]

    return y



def seasonalize(deTrended, m):

    length = len(deTrended)
    means = []

    # the for loop calculates the mean for each seasonal component
    for i in range(m):
        mean = deTrended[i:length:m].mean()
        means.append(mean)

    return means



def residual(y, dec_type):

    # calculate the residual component for the additive and the multiplicative dec_type correspondingly
    if(dec_type[0] == 'a'):
        y['R'] = y.loc[:, 'y'] - y.loc[:, 'T'] - y.loc[:, 'S']
    else:
        y['R'] = y.loc[:, 'y'] / y.loc[:, 'T'] / y.loc[:, 'S']

    return y



def decompose(y, m=7, dec_type='additive'):

    y = y.copy(deep=True)

    # call the above defined functions
    y = trend(y, m, dec_type)
    y = seasonality(y, m, dec_type)
    y = residual(y, dec_type)

    return y



def plot(y, title='Actual Data', xlab='time', ylab='value', dec_type='additive'):
    plt.style.use('ggplot')
    # plot the time series and all the components
    plt.subplot(4, 1, 1)
    plt.plot(y['t'], y['y'], color='darkblue')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([y['t'].min(), y['t'].max()])

    plt.subplot(4, 1, 2)
    plt.plot(y['t'], y['T'], color='darkblue')
    plt.title('Trend')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([y['t'].min(), y['t'].max()])

    plt.subplot(4, 1, 3)
    plt.plot(y['t'], y['S'], color='darkblue')
    plt.title('Seasonality')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([y['t'].min(), y['t'].max()])

    plt.subplot(4, 1, 4)
    if(dec_type[0] == 'a'):
        plt.stem(y['t'], y['R'], markerfmt=' ', linefmt='darkblue')
    else:
        plt.stem(y['t'], y['R'], markerfmt=' ', linefmt='darkblue', bottom=1)
    plt.title('Residual')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim([y['t'].min(), y['t'].max()])
    
    plt.subplots_adjust(hspace=0.75, top=0.956, bottom=0.064, left=0.136)

    plt.show()




