#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas Bury

Code to simulate the RM model and compute EWS

"""

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import EWS function
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute
from ews_spec import pspec_welch, pspec_metrics


#--------------------------------
# Adjustable parameters
#–-----------------------------


# Simulation parameters
dt = 0.1
t0 = 0
tmax = 200
tburn = 100 # burn-in period
numSims = 1
seed = 0 # random number generation seed

# EWS parameters
rw = 0.5 # rolling window
bw = 0.1 # bandwidth
lags = [1,2] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','aic'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 2 # offset for rolling window when doing spectrum metrics


#----------------------------------
# Simulate many (transient) realisations
#----------------------------------

# Model

def de_fun_x(x,y,r,k,a,h):
    return r*x*(1-x/k) - (a*x*y)/(1+a*h*x)

def de_fun_y(x,y,e,a,h,m):
    return e*a*x*y/(1+a*h*x) - m*y
    
# Model parameters
sigma_x = 0.01 # noise intensity
sigma_y = 0.01
r = 10
k = 1.7
h = 0.06
e = 0.5
m = 5
al = 25 # control parameter initial value
ah = 40 # control parameter final value
abif = 39.23 # bifurcation point (computed in Mathematica)
x0 = 1 # intial condition (equilibrium value computed in Mathematica)
y0 = 0.412




# initialise DataFrame for each variable to store all realisations
df_sims_x = pd.DataFrame([])
df_sims_y = pd.DataFrame([])

# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))
y = np.zeros(len(t))

# Set up control parameter a, that increases linearly in time from al to ah
a = pd.Series(np.linspace(al,ah,len(t)),index=t)
# Time at which bifurcation occurs
tbif = a[a > abif].index[1]

## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)


# loop over simulations
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
    dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))
    
    dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
    dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun_x(x0,y0,r,k,a[0],h)*dt + dW_x_burn[i]
        y0 = y0 + de_fun_y(x0,y0,e,a[0],h,m)*dt + dW_y_burn[i]
        
    # Initial condition post burn-in period
    x[0]=x0
    y[0]=y0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun_x(x[i],y[i],r,k,a.iloc[i],h)*dt + dW_x[i]
        y[i+1] = y[i] + de_fun_y(x[i],y[i],e,a.iloc[i],h,m)*dt + dW_y[i]
        # make sure that state variable remains >= 0 
        if x[i+1] < 0:
            x[i+1] = 0
        if y[i+1] < 0:
            y[i+1] = 0
            
    # Store data as a Series indexed by time
    series_x = pd.Series(x, index=t)
    series_y = pd.Series(y, index=t)
    
    # add Series to DataFrames of realisations
    df_sims_x['Sim '+str(j+1)] = series_x
    df_sims_y['Sim '+str(j+1)] = series_y




#----------------------
## Execute ews_compute for each realisation in x
#---------------------

# Sample from time-series at uniform intervals of width dt2
dt2 = 0.5
df_sims_filt = df_sims_x[np.remainder(df_sims_x.index,dt2) == 0]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []

# loop through each trajectory as an input to ews_compute
for i in range(numSims):
    df_temp = ews_compute(df_sims_filt['Sim '+str(i+1)], 
                      roll_window = rw, 
                      band_width = bw,
                      lag_times = lags, 
                      ews = ews,
                      ham_length = ham_length,
                      ham_offset = ham_offset,
                      pspec_roll_offset = pspec_roll_offset,
                      upto=tbif)
    # include a column in the dataframe for realisation number
    df_temp['Realisation number'] = pd.Series((i+1)*np.ones([len(t)],dtype=int),index=t)
    
    # add DataFrame to list
    appended_ews.append(df_temp)
    
    # print status every 10 realisations
    if np.remainder(i+1,1)==0:
        print('Realisation '+str(i+1)+' complete')


# concatenate EWS DataFrames - use realisation number and time as indices
df_ews = pd.concat(appended_ews).set_index('Realisation number',append=True).reorder_levels([1,0])


#
##------------------------
## Plots of EWS
##-----------------------
#
## plot of all variance trajectories
#df_ews.loc[:,'Variance'].unstack(level=0).plot(legend=False, title='Variance') # unstack puts index back as a column
#
## plot of all autocorrelation trajectories
#df_ews.loc[:,'Lag-1 AC'].unstack(level=0).plot(legend=False, title='Lag-1 AC') 
#
## plot of all smax trajectories
#df_ews.loc[:,'Smax'].unstack(level=0).dropna().plot(legend=False, title='Smax') # drop Nan values
#


# Make plot of EWS
plot_num = 1
fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews.loc[plot_num][['State variable','Smoothing']].plot(ax=axes[0],title='Early warning signals for the Consumer Resource Model')
df_ews.loc[plot_num]['Variance'].plot(ax=axes[1],legend=True)
df_ews.loc[plot_num][['Lag-1 AC','Lag-2 AC']].plot(ax=axes[1], secondary_y=True,legend=True)
df_ews.loc[plot_num]['Smax'].dropna().plot(ax=axes[2],legend=True)
df_ews.loc[plot_num][['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3],legend=True)





#-------------------------------------
# Display power spectrum and fits at a given instant in time
#------------------------------------

t_pspec = 175

# Use function pspec_welch to compute the power spectrum of the residuals at a particular time
pspec=pspec_welch(df_ews.loc[plot_num][t_pspec-rw*len(df_sims_filt.index):t_pspec]['Residuals'], dt2, ham_length=ham_length, w_cutoff=1)

# Execute the function pspec_metrics to compute the AIC weights and fitting parameters
spec_ews = pspec_metrics(pspec, ews=['smax', 'cf', 'aic', 'aic_params'])
# Define the power spectrum models
def fit_fold(w,sigma,lam):
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))
        
def fit_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))
        
def fit_null(w,sigma):
    return sigma**2/(2*np.pi)* w**0


# Make plot
w_vals = np.linspace(-max(pspec.index),max(pspec.index),100)

fig2=plt.figure(2)
pspec.plot(label='Measured')
plt.plot(w_vals, fit_fold(w_vals, spec_ews['Params fold']['sigma'], spec_ews['Params fold']['lam']),label='Fold (AIC='+str(round(spec_ews['AIC fold'],2))+')')
plt.plot(w_vals, fit_hopf(w_vals, spec_ews['Params hopf']['sigma'], spec_ews['Params hopf']['mu'], spec_ews['Params hopf']['w0']),label='Hopf (AIC='+str(round(spec_ews['AIC hopf'],2))+')')
plt.plot(w_vals, fit_null(w_vals, spec_ews['Params null']['sigma']),label='Null (AIC='+str(round(spec_ews['AIC null'],2))+')')
plt.ylabel('Power')
plt.legend()
plt.title('Power spectrum and fits at time t='+str(t_pspec))





