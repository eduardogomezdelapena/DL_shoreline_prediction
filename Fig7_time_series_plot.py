#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: EGP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from matplotlib.colors import LinearSegmentedColormap
import os
#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

#CVisionDeficiency friendly palette from  Crameri, F. (2018). 
#Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862
cm_data = np.loadtxt("./colormaps/roma.txt")
cvd_fmap = LinearSegmentedColormap.from_list('cvd_friendly', cm_data)

#Import Hybrid ensemble time series (test period)
mat_hyb= pd.read_csv('./Data/Hybrid_ensemble.csv')  
mat_hyb['Datetime'] = pd.to_datetime(mat_hyb['Datetime'])
mat_hyb = mat_hyb.set_index(['Datetime'])

#Import CNN ensemble time series (test period)
mat_cnn= pd.read_csv('./Data/CNN_ensemble.csv')  
mat_cnn['Datetime'] = pd.to_datetime(mat_cnn['Datetime'])
mat_cnn = mat_cnn.set_index(['Datetime'])

#Import Observed Shoreline
inputs= pd.read_csv('./Data/inputs_target.csv')  
inputs['Datetime'] = pd.to_datetime(inputs['Datetime'])
inputs = inputs.set_index(['Datetime'])

#Get dates intersection  between  ML ensemble and inputs
idx_intsct= inputs.index.intersection(mat_hyb.index)
inputs= inputs[idx_intsct [0]: idx_intsct[-1]]

#Envelopes
lower= mat_cnn.min(axis=1)
upper= mat_cnn.max(axis=1)
mean= mat_cnn.mean(axis=1)

lower_hyb= mat_hyb.min(axis=1)
upper_hyb= mat_hyb.max(axis=1)
mean_hyb= mat_hyb.mean(axis=1)
######################PLOT FIGURE 7############################################
fig,ax = plt.subplots(3,1,figsize=(11.8,6.8))
fs= 20
fs_tk= 14
fs_leg= 12
colors = cvd_fmap(np.linspace(0, 1, 4))

#Benchmarks
ax[0].plot(inputs.yout,'black')
ax[0].plot(mat_cnn.spads,color=colors[0],label=r'$SPADS$')
ax[0].plot(mat_cnn.shorefor,color=colors[1],label=r'$ShoreFor$')
ax[0].xaxis.set_major_locator(mdates.YearLocator())
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[0].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
ax[0].grid()
ax[0].legend(loc='lower right',fontsize=fs_leg)
ax[0].tick_params(axis='both', which='major', labelsize=fs_tk)

#CNN ensemble
ax[1].plot(inputs.yout,'black')
ax[1].fill_between(mat_cnn.index, y1=lower, y2=upper, 
                 alpha=0.3, color=colors[2])
ax[1].plot(mat_cnn.index, mean, alpha=0.9,
                  color=colors[2], label= "CNN")
ax[1].xaxis.set_major_locator(mdates.YearLocator())
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[1].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
ax[1].grid()
ax[1].legend(loc='lower right',fontsize=fs_leg)
ax[1].tick_params(axis='both', which='major', labelsize=fs_tk)

#CNN-LSTM ensemble
ax[2].plot(inputs.yout,'black')
ax[2].fill_between(mat_hyb.index, y1=lower_hyb, y2=upper_hyb, 
                 alpha=0.3, color=colors[3])
ax[2].plot(mat_hyb.index, mean_hyb, alpha=0.9,
                  color=colors[3], label= "CNN-LSTM")
ax[2].xaxis.set_major_locator(mdates.YearLocator())
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax[2].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
ax[2].grid()
ax[2].legend(loc='lower right',fontsize=fs_leg)
ax[2].tick_params(axis='both', which='major', labelsize=fs_tk)

plt.rcParams.update({'font.size': 28})
fig.text(0.04, 0.5, 'Cross-shore displacement ' r'$[m]$', va='center',
         rotation='vertical',fontsize=fs)
#Uncomment to save plot
# plt.savefig('./Figures/Fig7.png',
#             bbox_inches='tight',dpi=300)