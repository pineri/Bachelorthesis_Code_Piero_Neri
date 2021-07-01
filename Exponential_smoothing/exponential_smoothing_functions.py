# imports

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import math
import copy


def AIC(yy, y_fit, T, k):
    
    yy = np.array(yy)
    y_fit = np.array(y_fit)
    
        
    SSE = ((yy - y_fit)**2).sum()
    
    
    AIC_SCORE = T*math.log(SSE/T) + 2*k
    
        
    return AIC_SCORE


# # N-N method

def N_N(yy, alpha, l, h=1):
    
    # fitting section    
    l_t = l
    fit = []
        
    for t in range(1, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        l_t = alpha*yyy + (1 - alpha)*l_t_1
        
        fit.append(l_t_1)
    
    # forecasting section 
    forecasts = [l_t for H in range(1, h+1)]
    
    return (fit, forecasts)

def auto_N_N(data, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[:, 'y'].mean())

    
    # experimental values 
    yy = [None] * (y.shape[0] + 1)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[1:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 l_0
    k = 2
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
                
        # perform exponential smoothing
        (fit, forecasts) = N_N(yy, alpha, l, h)

        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
        if(min_AIC == 'start' or AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_fit = fit
            best_forecasts = forecasts
    
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        
        # perform exponential smoothing
        (fit, forecasts) = N_N(yy, alpha, l, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_fit = fit
            best_forecasts = forecasts
    
    
    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha)


# # N-A Method

def N_A(yy, m, alpha, gamma, l, s, h=1):
    
    # fitting section    
    l_t = l
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy - s_t_m) + (1 - alpha)*l_t_1
        s[t] = gamma*(yyy - l_t_1) + (1 - gamma)*s_t_m
        
        fit.append(l_t_1 + s_t_m)
    
    # forecasting section 
    t = len(yy)-1

    forecasts = [l_t + s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)

def auto_N_A(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) - A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 gamma, 1 l_0, m s_i
    k = 3 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_gamma = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
            gamma = float(gamma)
                
            # perform exponential smoothing
            (fit, forecasts) = N_A(yy, m, alpha, gamma, l, s, h)

            # calculate the AIC 
            AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
            if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                min_AIC = AIC_SCORE
                best_alpha = alpha
                best_gamma = gamma
                best_fit = fit
                best_forecasts = forecasts
    
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        gamma = float(np.random.choice(gamma_space))
        
        # perform exponential smoothing
        (fit, forecasts) = N_A(yy, m, alpha, gamma, l, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_gamma = gamma
            best_fit = fit
            best_forecasts = forecasts
    
    
    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_gamma)


# # N-M method

def N_M(yy, m, alpha, gamma, l, s, h=1):
    
    # fitting section    
    l_t = l
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy / s_t_m) + (1 - alpha)*l_t_1
        s[t] = gamma*(yyy / l_t_1) + (1 - gamma)*s_t_m
        
        fit.append(l_t_1 * s_t_m)
    
    # forecasting section 
    t = len(yy)-1

    forecasts = [l_t * s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)

def auto_N_M(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) / A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 gamma, 1 l_0, m s_i
    k = 3 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_gamma = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
            gamma = float(gamma)
                
            # perform exponential smoothing
            (fit, forecasts) = N_M(yy, m, alpha, gamma, l, s, h)

            # calculate the AIC 
            AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
            if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                min_AIC = AIC_SCORE
                best_alpha = alpha
                best_gamma = gamma
                best_fit = fit
                best_forecasts = forecasts
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        gamma = float(np.random.choice(gamma_space))
        
        # perform exponential smoothing
        (fit, forecasts) = N_M(yy, m, alpha, gamma, l, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_gamma = gamma
            best_fit = fit
            best_forecasts = forecasts
    
    
    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_gamma)


# # A-N Method

def A_N(yy, alpha, beta, l, b, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(1, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        l_t = alpha*yyy + (1 - alpha)*(l_t_1 + b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*b_t_1
        
        fit.append(l_t_1 + b_t_1)
    
    # forecasting section 
    forecasts = [l_t + H*b_t for H in range(1, h+1)]
        
    
    return (fit, forecasts)

def auto_A_N(data, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    b = float((y.loc[y.shape[0]-1, 'y'] - y.loc[0, 'y'])/(y.shape[0] - 1))


    # experimental values 
    yy = [None] * (y.shape[0]+1)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[1:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 l_0, 1 b_0
    k = 4
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
                
            # perform exponential smoothing
            (fit, forecasts) = A_N(yy, alpha, beta, l, b, h)

            # calculate the AIC 
            AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
            if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                min_AIC = AIC_SCORE
                best_alpha = alpha
                best_beta = beta
                best_fit = fit
                best_forecasts = forecasts
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        
        # perform exponential smoothing
        (fit, forecasts) = A_N(yy, alpha, beta, l, b, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_fit = fit
            best_forecasts = forecasts
    
    
    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta)


# # A-A Method

def A_A(yy, m, alpha, beta, gamma, l, b, s, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy - s_t_m) + (1 - alpha)*(l_t_1 + b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*b_t_1
        s[t] = gamma*(yyy - l_t_1 - b_t_1) + (1 - gamma)*s_t_m
        
        fit.append(l_t_1 + b_t_1 + s_t_m)
    
    # forecasting section 
    t = len(yy)-1

    forecasts = [l_t + H*b_t + s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)

def auto_A_A(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    bs = np.array([(y.loc[i, 'y'] - y.loc[i-m+1, 'y'])/(m-1) for i in range(m-1, y.shape[0])])
    b = float(bs.mean())

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) - A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp

    
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 gamma, 1 l_0, 1 b_0, m s_i
    k = 5 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_gamma = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
            for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
                gamma = float(gamma)
                
                # perform exponential smoothing
                (fit, forecasts) = A_A(yy, m, alpha, beta, gamma, l, b, s, h)

                # calculate the AIC 
                AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
                if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                    min_AIC = AIC_SCORE
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
                    best_fit = fit
                    best_forecasts = forecasts
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        gamma = float(np.random.choice(gamma_space))
        
        # perform exponential smoothing
        (fit, forecasts) = A_A(yy, m, alpha, beta, gamma, l, b, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_fit = fit
            best_forecasts = forecasts
    
    

    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta, best_gamma)


# # A-M method

def A_M(yy, m, alpha, beta, gamma, l, b, s, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy / s_t_m) + (1 - alpha)*(l_t_1 + b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*b_t_1
        s[t] = gamma*(yyy / (l_t_1 + b_t_1)) + (1 - gamma)*s_t_m
        
        fit.append((l_t_1 + b_t_1) * s_t_m)
    
    # forecasting section 
    t = len(yy)-1

    forecasts = [(l_t + H*b_t) * s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)

def auto_A_M(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    bs = np.array([(y.loc[i, 'y'] - y.loc[i-m+1, 'y'])/(m-1) for i in range(m-1, y.shape[0])])
    b = float(bs.mean())
    

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) / A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 gamma, 1 l_0, 1 b_0, m s_i
    k = 5 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_gamma = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
            for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
                gamma = float(gamma)
                
                # perform exponential smoothing
                (fit, forecasts) = A_M(yy, m, alpha, beta, gamma, l, b, s, h)

                # calculate the AIC 
                AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
                if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                    min_AIC = AIC_SCORE
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
                    best_fit = fit
                    best_forecasts = forecasts
    
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        gamma = float(np.random.choice(gamma_space))
        
        # perform exponential smoothing
        (fit, forecasts) = A_M(yy, m, alpha, beta, gamma, l, b, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_fit = fit
            best_forecasts = forecasts
    

    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta, best_gamma)


# #Ad-N Method

def Ad_N(yy, alpha, beta, phi, l, b, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(1, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        l_t = alpha*yyy + (1 - alpha)*(l_t_1 + phi*b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*phi*b_t_1
        
        fit.append(l_t_1 + phi*b_t_1)
    
    # forecasting section
    phi_h = np.array([phi**i for i in range(1, h + 1)])
    phi_h = np.cumsum(phi_h)
    
    forecasts = [l_t + phi_h[i]*b_t for i in range(0, h)]

    
    return (fit, forecasts)


def auto_Ad_N(data, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    b = float((y.loc[y.shape[0]-1, 'y'] - y.loc[0, 'y'])/(y.shape[0] - 1))


    # experimental values 
    yy = [None] * (y.shape[0]+1)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[1:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 l_0, 1 b_0
    k = 4
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_phi = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
            for phi in np.linspace(0.8, 0.98, initial_grid):
                phi = float(phi)
                
                # perform exponential smoothing
                (fit, forecasts) = Ad_N(yy, alpha, beta, phi, l, b, h)

                # calculate the AIC 
                AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
                if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                    min_AIC = AIC_SCORE
                    best_alpha = alpha
                    best_beta = beta
                    best_phi = phi
                    best_fit = fit
                    best_forecasts = forecasts
    
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    surr_phi = 0.18/(2*(initial_grid-1))
    
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])
    phi_space = np.array([el for el in list(np.linspace(best_phi-surr_phi, best_phi+surr_phi, n)) if 0.8<=el and 0.98>=el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        phi = float(np.random.choice(phi_space))
        
        # perform exponential smoothing
        (fit, forecasts) = Ad_N(yy, alpha, beta, phi, l, b, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_phi = phi
            best_fit = fit
            best_forecasts = forecasts
    
    
    
    
    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta, best_phi)


# # Ad-A Method

def Ad_A(yy, m, alpha, beta, gamma, phi, l, b, s, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy - s_t_m) + (1 - alpha)*(l_t_1 + phi*b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*phi*b_t_1
        s[t] = gamma*(yyy - l_t_1 - phi*b_t_1) + (1 - gamma)*s_t_m
        
        fit.append(l_t_1 + phi*b_t_1 + s_t_m)
    
    # forecasting section 
    t = len(yy)-1
    
    phi_h = np.array([phi**i for i in range(1, h + 1)])
    phi_h = np.cumsum(phi_h)

    forecasts = [l_t + phi_h[H-1]*b_t + s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)


def auto_Ad_A(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    bs = np.array([(y.loc[i, 'y'] - y.loc[i-m+1, 'y'])/(m-1) for i in range(m-1, y.shape[0])])
    b = float(bs.mean())
    

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) - A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 gamma, 1 l_0, 1 b_0, m s_i
    k = 5 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_gamma = 0
    best_phi = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
            for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
                gamma = float(gamma)
                for phi in np.linspace(0.8, 0.98, initial_grid):
                    phi = float(phi)
                
                    # perform exponential smoothing
                    (fit, forecasts) = Ad_A(yy, m, alpha, beta, gamma, phi, l, b, s, h)

                    # calculate the AIC 
                    AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
                    if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                        min_AIC = AIC_SCORE
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                        best_phi = phi
                        best_fit = fit
                        best_forecasts = forecasts
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    surr_phi = 0.18/(2*(initial_grid-1))
    
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])
    phi_space = np.array([el for el in list(np.linspace(best_phi-surr_phi, best_phi+surr_phi, n)) if 0.8<=el and 0.98>=el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        gamma = float(np.random.choice(gamma_space))
        phi = float(np.random.choice(phi_space))
        
        # perform exponential smoothing
        (fit, forecasts) = Ad_A(yy, m, alpha, beta, gamma, phi, l, b, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_phi = phi
            best_fit = fit
            best_forecasts = forecasts
    

    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta, best_gamma, best_phi)


# # Ad-M Method

def Ad_M(yy, m, alpha, beta, gamma, phi, l, b, s, h=1):
    
    # fitting section    
    l_t = l
    b_t = b
    fit = []
        
    for t in range(m, len(yy)):
        yyy = yy[t]
        l_t_1 = l_t
        b_t_1 = b_t
        s_t_m = s[t-m]
        l_t = alpha*(yyy / s_t_m) + (1 - alpha)*(l_t_1 + phi*b_t_1)
        b_t = beta*(l_t - l_t_1) + (1 - beta)*phi*b_t_1
        s[t] = gamma*(yyy / (l_t_1 + phi*b_t_1)) + (1 - gamma)*s_t_m
        
        fit.append((l_t_1 + phi*b_t_1) * s_t_m)
    
    # forecasting section 
    t = len(yy)-1
    
    phi_h = np.array([phi**i for i in range(1, h + 1)])
    phi_h = np.cumsum(phi_h)

    forecasts = [(l_t + phi_h[H-1]*b_t) * s[t+H-m*(int((H-1)/m)+1)] for H in range(1, h+1)]
        
    
    return (fit, forecasts)


def auto_Ad_M(data, m, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    y = data.copy(deep=True)                        
    y.columns = ['t', 'y']
    
    
    # initial level
    l = float(y.loc[0, 'y'])
    
    # initial trend
    bs = np.array([(y.loc[i, 'y'] - y.loc[i-m+1, 'y'])/(m-1) for i in range(m-1, y.shape[0])])
    b = float(bs.mean())
    

    # initial seasonal indices
    s = [None] * (y.shape[0]+m) 
    n_periods = y.shape[0] // m
    A = [y.loc[i*m:((i+1)*m)-1, 'y'].mean() for i in range(n_periods)]
    A = np.array(A)
    A = np.repeat(A,m)
    seas = np.array(y.loc[:(n_periods*m)-1, 'y']) / A
    seas = seas.reshape(n_periods, m)
    seas_mean = seas.mean(axis=0)
    s_temp = [float(i) for i in seas_mean]
    s[0:m] = s_temp
    

    # experimental values 
    yy = [None] * (y.shape[0]+m)
    yy_temp = list(y['y'])
    yy_temp = [float(i) for i in yy_temp]
    yy[m:] = yy_temp
    

    T = y.shape[0]
    
    # 1 alpha, 1 beta, 1 gamma, 1 l_0, 1 b_0, m s_i
    k = 5 + m
    
    # store the best results   
    min_AIC = 'start' 
    best_alpha = 0
    best_beta = 0
    best_gamma = 0
    best_phi = 0
    best_fit = None
    best_forecasts = None

    # grid search for optimal parameters
    for alpha in np.linspace(0.00001, 0.99999, initial_grid):
        alpha = float(alpha)
        for beta in np.linspace(0.00001, 0.99999, initial_grid):
            beta = float(beta)
            for gamma in np.linspace(0.00001, alpha-0.00001, initial_grid):
                gamma = float(gamma)
                for phi in np.linspace(0.8, 0.98, initial_grid):
                    phi = float(phi)
                
                    # perform exponential smoothing
                    (fit, forecasts) = Ad_M(yy, m, alpha, beta, gamma, phi, l, b, s, h)

                    # calculate the AIC 
                    AIC_SCORE = AIC(yy[-T:], fit, T, k)
            
            
                    if(min_AIC == 'start' or AIC_SCORE < min_AIC):
                        min_AIC = AIC_SCORE
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                        best_phi = phi
                        best_fit = fit
                        best_forecasts = forecasts
    
    
    n = fine_grid
    surr = 1/(2*(initial_grid-1))
    
    surr_phi = 0.18/(2*(initial_grid-1))
    
    
    alpha_space = np.array([el for el in list(np.linspace(best_alpha-surr, best_alpha+surr, n)) if 0<el and 1>el])
    beta_space = np.array([el for el in list(np.linspace(best_beta-surr, best_beta+surr, n)) if 0<el and 1>el])
    gamma_space = np.array([el for el in list(np.linspace(best_gamma-surr, best_gamma+surr, n)) if 0<el and 1>el])
    phi_space = np.array([el for el in list(np.linspace(best_phi-surr_phi, best_phi+surr_phi, n)) if 0.8<=el and 0.98>=el])

    
    for i in range(rand_iter):

        alpha = float(np.random.choice(alpha_space))
        beta = float(np.random.choice(beta_space))
        gamma = float(np.random.choice(gamma_space))
        phi = float(np.random.choice(phi_space))
        
        # perform exponential smoothing
        (fit, forecasts) = Ad_M(yy, m, alpha, beta, gamma, phi, l, b, s, h)
        
        # calculate the AIC 
        AIC_SCORE = AIC(yy[-T:], fit, T, k)
        
        if(AIC_SCORE < min_AIC):
            min_AIC = AIC_SCORE
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
            best_phi = phi
            best_fit = fit
            best_forecasts = forecasts
    
    
    

    # data frame with fitted values
    fit = pd.DataFrame()
    fit['t'] = y['t']
    fit['y'] = best_fit
    
    # data frame with point-forecast
    forecasts = pd.DataFrame()
    forecasts['t'] = pd.Series([y.loc[y.shape[0]-1, 't'] + H*(y.loc[y.shape[0]-1, 't'] - y.loc[y.shape[0]-2, 't']) for H in range(1, h+1)])
    forecasts['y'] = best_forecasts
                
    return (fit, forecasts, min_AIC, best_alpha, best_beta, best_gamma, best_phi)


# # Evaluation

def rolling_forecasting_origin_evaluation(training_data, test_data, method , m=0, h=1, initial_grid=6, fine_grid=11, rand_iter=500):
    
    training_data.columns = ['t', 'y']
    test_data.columns = ['t', 'y']
    

    errors = []
    
    assert(h <= test_data.shape[0])

    for i in range(h-1, test_data.shape[0]):
        
        if(m == 0):
            result = method(training_data, h, initial_grid, fine_grid, rand_iter)
        else:
            result = method(training_data, m, h, initial_grid, fine_grid, rand_iter)
         
        errors.append(list(np.array(test_data['y'])[i-h+1:i+1] - np.array(result[1]['y'])))
    
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




