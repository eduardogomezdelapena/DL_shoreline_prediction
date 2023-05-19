#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:42:19 2023

@author: egom802
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
import csv 



from datetime import date
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import math
import numpy as np
import string



#%%
site_str="Tairua"

rootdir='/home/egom802/Dropbox (Uni of Auckland)/Documents/PhD_UoA/ML_paper/'
# rootdir='/home/eduardo/Documents/PhD_UoA/ML_paper/'

gendir= rootdir+ site_str
dir_csv= rootdir+'Codes/Python/Architectures/Hyp_search/Metrics/'

sens_dir="Driver_sens/"

#Define Error_Architecture string
erar_str= "Wavesonly_Mielke_Hybrid"
# erar_str= "MSE_CNN"

imagedir=gendir+'Codes/Python/Images'
os.chdir(gendir + '/Original_PC_wave_shoreline_csv')

#Choose which inputs will go to the model
# pc_bool=False
# waves_bool=True
# ar_bool=False
PC_number= 10
trial_number= str(1)
trial_str= trial_number+"d"; title_str= site_str+"_Waves_"+str(PC_number)+"PC"

#%%###############################################################################
#Import csv files, convert to datetime 

mat_pc= pd.read_csv('matrix_time_first100PC.csv')  

key_list= ['PC('+str(x)+')' for x in range(1,PC_number+1)] #Create PC strings
key_list= ["Y","M","D","h","m","s"] + key_list #Append time columns
mat_pc= mat_pc[key_list] #Obtain desireed PCs + time columns


mat_waves=  pd.read_csv('matrix_time_Hs_Tp_Dir.csv')
mat_out= pd.read_csv('matrix_time_outputy.csv')


def columns_to_datetime(mat):
    """
    Convert date columns (Y,M,D,h,m,s) of imported dataframe to datetime
    """
    mat['Datetime']  = pd.to_datetime(mat['Y'].astype(str) + '-' +
                                  mat['M'].astype(str) + '-' +
                                  mat['D'].astype(str) + ' ' +
                                  mat['h'].astype(str) + ':' +
                                  mat['m'].astype(str) + ':' +
                                  mat['s'].astype(str))

    mat= mat.drop(['Y','M','D', 'h','m','s'], axis=1)
    mat = mat.set_index('Datetime')
    return mat

#Convert date columns to datetime 
mat_pc= columns_to_datetime(mat_pc)
mat_waves= columns_to_datetime(mat_waves)
mat_out= columns_to_datetime(mat_out)  

#%%#######################################################Shift?????###############
mat_out=mat_out.shift(periods=-3) 
#%%#
#SMoothed time series  
mat_out=mat_out.rolling(7).mean() 
#Shift y observed -1 day
#ar=mat_out.shift(1, freq='D') 
W_dir=mat_waves["Dir"]   
#%%############################################################################
#Wind Direction (degrees) transform to vector
#Obtain wave speed
#First calculate wave length for deep water
mat_waves["WL"]= (9.8 * np.power(mat_waves.Tp,2)) / (2*math.pi) 
#Wave_velocity= 1/Tp * Wave_length
mat_waves["Wv"]= (1/mat_waves.Tp) * mat_waves.pop('WL')

#Transform to x,y components
wv = mat_waves.pop('Wv')
# Convert to radians.
wd_rad = mat_waves.pop('Dir')*np.pi / 180
# Calculate the wind x and y components.
mat_waves['Wvx'] = wv*np.cos(wd_rad)
mat_waves['Wvy'] = wv*np.sin(wd_rad)
#%%############################################################################
######################CONCATENATE INPUTS#######################################
#Here we choose which inputs go to the NN
# sum_bool= sum([mat_pc, mat_waves,ar])

inputs = [mat_pc, mat_waves, mat_out]
         
#%%#######################HOMOGENEOUS SERIES###################################
# Dropnans, resample and get values with same dates
#Resample all series to have daily average
mat_in= pd.concat(inputs).resample('D').mean()

W_dir=W_dir.resample('D').mean()
#Drop nan values to have same length
mat_in.dropna(inplace=True)             
#%%
lw=0.7
ms= 0.9

fig, axs = plt.subplots(4,figsize=(8.5,6.5))


# for n, ax in enumerate(axs):
for n, ax in enumerate(axs[0:2]):
    
    ax.text(-0.15, 0.9, string.ascii_lowercase[n] + ")", transform=ax.transAxes, 
            size=20, weight='bold')
    
    
axs[0].plot(mat_in['yout'],lw=lw,color='k')
axs[0].set_ylabel("cross-shore\n displacement [m]")

axs[1].plot(mat_in['Hs'],lw=lw,color='k')
axs[1].set_ylabel("$H_{s}$ [m]")

axs[2].plot(mat_in['Tp'],lw=lw,color='k')
axs[2].set_ylabel("$T_{p}$ [s]")

axs[3].plot(W_dir,'.',ms=ms,color='k')
axs[3].set_ylabel(r'$\theta \ $''[$^{\circ}$]')

plt.setp(axs, xlim=[date(1999, 1, 8), date(2016, 12, 30)])


plt.savefig('/home/egom802/Dropbox (Uni of Auckland)/Documents/PhD_UoA/ML_paper/Figures/Fig4_waveinputs.png',
            bbox_inches='tight', format='png', dpi=300)









