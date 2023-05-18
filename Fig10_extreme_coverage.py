#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: EGP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os
#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

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

#Create common dataframe and plot
yresults= pd.DataFrame(mat_hyb.spads,index=mat_hyb.index,columns=["hyb_m"])
yresults["spads"]=mat_hyb.spads; yresults["shorefor"]=mat_hyb.shorefor
yresults["hyb_m"]=mat_hyb.mean(axis=1)
yresults["hyb_u"]=mat_hyb.max(axis=1)
yresults["hyb_l"]=mat_hyb.min(axis=1)
yresults["obs"]= inputs.yout
######## Count observations within the shaded area, shift +- 1 day#############
#Set the number of days to shift
n_shift= 0

#Observations that are within the original shade
obs_shade_bool= yresults["obs"].gt(
    yresults["hyb_m"]) & yresults["obs"].lt(yresults["hyb_u"])
obs_shade= yresults[obs_shade_bool.values]

#Shift time series + 1 day
obs_shift=yresults.shift(n_shift, freq='D')
obs_shift_bool= yresults["obs"].gt(
    obs_shift["hyb_l"]) & yresults["obs"].lt(obs_shift["hyb_u"])

#Shift time series - 1 day
obs_shift_neg=yresults.shift(-n_shift, freq='D')
obs_shift_bool_neg= yresults["obs"].gt(
    obs_shift_neg["hyb_l"]) & yresults["obs"].lt(obs_shift_neg["hyb_u"])

#Get dates intersection  between original and extended time series
idx_intsct= obs_shade_bool.index.intersection(obs_shift_bool.index)
#Cut dataframe by intersected dates
obs_shift_bool= obs_shift_bool[idx_intsct]
obs_shift_bool_neg= obs_shift_bool_neg[idx_intsct]

#Combine booleans
bool_comb= np.logical_or(obs_shift_bool_neg.values, obs_shift_bool.values)
bool_comb= np.logical_or(bool_comb,obs_shade_bool.values)

#Apply boolean to filter observations that lay +-1 day shade
or_shift= yresults[bool_comb]

#Percentage of observations 
obs_perc=len(or_shift)/len(yresults)*100
###############################################################################
#Plot the original + shifted 
#For plotting aesthetics
color= "lightskyblue";alpha= 1;fs= 20
fs_tk= 12;fs_leg= 11;ms= 4

fig,ax = plt.subplots(figsize=(11.8,4))

plt.fill_between(yresults.index, y1=yresults["hyb_l"], y2=yresults["hyb_u"], 
                 alpha=alpha, color=color, label="Min-max envelope")
plt.plot(yresults["obs"],'.k',ms=ms,label= str( 100- round(obs_perc,1)) + "%" )
plt.plot(or_shift["obs"],'.r',ms=ms, label= str( round(obs_perc,1)) + "%"  )
plt.legend()

ax.set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
ax.legend(loc='lower right',fontsize=fs_leg)
ax.tick_params(axis='both', which='major', labelsize=fs_tk)
# Uncomment to save plot
# plt.savefig('./Figures/Fig10.png',
#             bbox_inches='tight',dpi=300)