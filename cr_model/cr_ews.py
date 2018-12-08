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
import seaborn as sns
import os

# import EWS function
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute


#---------------------
# Directory for data output
#–----------------------

# Name of directory within data_export
dir_name = 'cr_ews_1'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 0.02
t0 = 0
tmax = 200
tburn = 100 # burn-in period
numSims = 2
seed = 0 # random number generation seed

# EWS parameters
dt2 = 0.5 # spacing between time-series for EWS computation
rw = 0.25 # rolling window
bw = 0.1 # bandwidth
lags = [1,2,4] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','aic','cf'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 10 # offset for rolling window when doing spectrum metrics


#----------------------------------
# Simulate many (transient) realisations
#----------------------------------

# Model

def de_fun_x(x,y,r,k,a,h):
    return r*x*(1-x/k) - (a*x*y)/(1+a*h*x)

def de_fun_y(x,y,e,a,h,m):
    return e*a*x*y/(1+a*h*x) - m*y
    
# Model parameters
sigma_x = 0.02 # noise intensity
sigma_y = 0.02
r = 10
k = 1.7
h = 0.06
e = 0.5
m = 5
al = 25 # control parameter initial value
ah = 42 # control parameter final value
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
print('\nBegin simulations \n')
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
    
    

    print('Simulation '+str(j+1)+' complete')


#----------------------
## Execute ews_compute for each realisation in x
#---------------------

# Filter time-series to have time-spacing dt2
df_sims_filt = df_sims_x.loc[::int(dt2/dt)]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []
appended_pspec = []

# loop through each trajectory as an input to ews_compute
print('\nBegin EWS computation\n')
for i in range(numSims):
    ews_dic = ews_compute(df_sims_filt['Sim '+str(i+1)], 
                      roll_window = rw, 
                      band_width = bw,
                      lag_times = lags, 
                      ews = ews,
                      ham_length = ham_length,
                      ham_offset = ham_offset,
                      pspec_roll_offset = pspec_roll_offset,
                      upto=tbif)
    # The DataFrame of EWS
    df_ews_temp = ews_dic['EWS metrics']
    # The DataFrame of power spectra
    df_pspec_temp = ews_dic['Power spectrum']
    
    # Include a column in the DataFrames for realisation number
    df_ews_temp['Realisation number'] = (i+1)*np.ones([len(df_ews_temp)],dtype=int)
    df_pspec_temp['Realisation number'] = (i+1)*np.ones([len(df_pspec_temp)],dtype=int)
    
    
    # Add DataFrames to list
    appended_ews.append(df_ews_temp)
    appended_pspec.append(df_pspec_temp)
    
    # Print status every realisation
    if np.remainder(i+1,1)==0:
        print('EWS for realisation '+str(i+1)+' complete')


# Concatenate EWS DataFrames - use realisation number and time (and frequency for df_pspec) as indices
df_ews = pd.concat(appended_ews).set_index('Realisation number',append=True).reorder_levels([1,0])
df_pspec = pd.concat(appended_pspec).set_index('Realisation number',append=True).reorder_levels([2,0,1])


## export the first 5 realisations for plotting
#df_ews.loc[1:5].to_csv('data_export/'+dir_name+'/ews_singles.csv')


# Compute summary statistics of EWS over all realisations (mean, pm1 s.d.)
df_ews_means = df_ews[['Variance', 'Lag-1 AC', 'Lag-2 AC', 'Lag-4 AC', 
                      'AIC fold', 'AIC hopf']].mean(level='Time')
df_ews_deviations = df_ews[['Variance', 'Lag-1 AC', 'Lag-2 AC', 'Lag-4 AC', 
                      'AIC fold', 'AIC hopf']].std(level='Time')


## Export summary statistics for plotting in MMA
#df_ews_means.to_csv('data_export/'+dir_name+'/ews_mean.csv')
#df_ews_deviations.to_csv('data_export/'+dir_name+'/ews_std.csv')


#-------------------------
# Plots to visualise EWS
#-------------------------

# Realisation number to plot
plot_num = 1

## Plot of trajectory, smoothing and EWS
fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews.loc[plot_num][['State variable','Smoothing']].plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num]['Variance'].plot(ax=axes[1],legend=True)
df_ews.loc[plot_num][['Lag-1 AC','Lag-2 AC','Lag-4 AC']].plot(ax=axes[1], secondary_y=True,legend=True)
df_ews.loc[plot_num]['Smax'].dropna().plot(ax=axes[2],legend=True)
df_ews.loc[plot_num]['Coherence factor'].dropna().plot(ax=axes[2], secondary_y=True, legend=True)
df_ews.loc[plot_num][['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3],legend=True)


## Grid plot for evolution of the power spectrum in time
g = sns.FacetGrid(df_pspec.loc[plot_num][::2].reset_index(), 
                  col='Time',
                  col_wrap=3,
                  sharey=False,
                  aspect=1.5,
                  size=1.8
                  )

g.map(plt.plot, 'Frequency', 'Empirical', color='k', linewidth=2)
g.map(plt.plot, 'Frequency', 'Fit fold', color='b', linestyle='dashed', linewidth=1)
g.map(plt.plot, 'Frequency', 'Fit hopf', color='r', linestyle='dashed', linewidth=1)
g.map(plt.plot, 'Frequency', 'Fit null', color='g', linestyle='dashed', linewidth=1)
# Axes properties
axes = g.axes
# Set y labels
for ax in axes[::3]:
    ax.set_ylabel('Power')
# Set y limit as max power over all time
for ax in axes:
    ax.set_ylim(top=1.05*max(df_pspec.loc[plot_num]['Empirical']), bottom=0)

# Export figure
g.savefig('figures/pspec_evol1.png', dpi=200)











