#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:46:56 2023

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

from tsmoothie.smoother import *
from tsmoothie.bootstrap import BootstrappingWrapper
from tsmoothie.utils_func import create_windows, sim_seasonal_data, sim_randomwalk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn import preprocessing


import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import math
import numpy as np
from sklearn.metrics import mean_squared_error

#%%
site_str="Tairua"

rootdir='/home/egom802/Dropbox (Uni of Auckland)/Documents/PhD_UoA/ML_paper/'
# rootdir='/home/eduardo/Documents/PhD_UoA/ML_paper/'

gendir= rootdir+ site_str
dir_csv= rootdir+'Codes/Python/Architectures/Hyp_search/Metrics/'

sens_dir="Driver_sens/"

#Define Error_Architecture string
erar_str= "Wavesonly_Mielke_CNN"
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

#%%############################################################################
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

#%%#######################################################Shift?????###########
mat_out=mat_out.shift(periods=-3) 
#%%#
#SMoothed time series  
mat_out=mat_out.rolling(7).mean() 
#Shift y observed -1 day
#ar=mat_out.shift(1, freq='D')    
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

#inputs = [mat_pc, mat_waves, mat_out]
inputs = [mat_waves, mat_out]
         
#%%#######################HOMOGENEOUS SERIES###################################
# Dropnans, resample and get values with same dates
#Resample all series to have daily average
mat_in= pd.concat(inputs).resample('D').mean()
#Drop nan values to have same length
mat_in.dropna(inplace=True)             
                 
#Get dates intersection   
idx_intsct= mat_out.index.intersection(mat_in.index)
#Slice Matrix inputs with the intersect dates      
mat_in= mat_in[idx_intsct [0]: idx_intsct[-1]]
#Slice Matrix outputs with the intersect dates                
mat_out= mat_out[idx_intsct [0]: idx_intsct[-1]]   
#%%############################################################################
#Import SPADS, Shorefor csv files, convert to datetime 

mat_spads= pd.read_csv('./Data/Originals/matrix_time_spads_shorefor_obs.csv') 

#Convert date columns to datetime 
mat_spads= columns_to_datetime(mat_spads)

#Becareful with this data, as in Tairua the forecast is from July 2014 
#previous data is training in SPADS and ShoreFor

date_forecast= '2014-07-01'

mat_spads = mat_spads[date_forecast:mat_spads.index[-1]]

#Make sure all time -series end at the same date
#Get dates intersection with mat_spads
idx_intsct= mat_spads.index.intersection(mat_in.index)

#Slice Matrix inputs with the intersect dates      
mat_in= mat_in[mat_in.index[0]: idx_intsct[-1]]
#Slice Matrix outputs with the intersect dates                
mat_out= mat_out[mat_out.index[0]: idx_intsct[-1]]   

mat_in.to_csv(rootdir+'Codes_forsubmission/Data/inputs_target.csv')

#%%##################TRAIN, DEV AND TEST SPLITS################################

#Manually split by date 

#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))

#%%############################################################################
#########################DATA NORMALIZATION####################################
#Data normalization [-1,1] 
def normal_data(df,date):
    """
    Normalize dataframe values to range [-1,1]
    """
    #x= df.values
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
    #Here we fit the scaler 
    scaler= scaler.fit(df[df.index[0]:date])
    #Here we transform the data
    x_scaled = scaler.transform(df)
    df = pd.DataFrame(x_scaled,index=df.index, columns=df.columns)
    return scaler,df

_, mat_in_norm = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)


#%%##################TRAIN, DEV AND TEST SPLITS################################

train = mat_in_norm[mat_in_norm.index[0]:train_date].values.astype('float32')

#Development set (2 years before the test set)
devinit_date=pd.to_datetime(train_date) + timedelta(days=1)
devinit_date= str(devinit_date.strftime("%Y-%m-%d"))
dev_date=pd.to_datetime(date_forecast) - timedelta(days=1)
dev_date= str(dev_date.strftime("%Y-%m-%d"))
dev= mat_in_norm[devinit_date:dev_date].values.astype('float32')

#Test set, depends on study site
test = mat_in_norm[date_forecast:mat_in_norm.index[-1]].values.astype('float32')

##############################################################################

#%%############################################################################

#From pandas to array, HERE WE SEPARATE THE INPUTS FROM THE Y_OUTPUT

# split a multivariate sequence into samples for LSTM
# How much steps can the ANN look backward

def split_sequences(sequences, n_steps_in, n_steps_out):
     X, y = list(), list()
     for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
             break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
     return np.array(X), np.array(y)

#how many time steps are taken into account,
#how many time steps is the network allowed to look back
n_steps_in, n_steps_out =40,1


train_x, train_y = split_sequences(train, n_steps_in, n_steps_out)

dev_x, dev_y = split_sequences(dev, n_steps_in, n_steps_out)

test_x, test_y = split_sequences(test, n_steps_in, n_steps_out)
# # the dataset knows the number of features, e.g. 2

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

#%% Import SPADS and ShoreFor Data


plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))

plot_train = pd.to_datetime(train_date) - timedelta(days=n_steps_in-1)
plot_train= str(plot_train.strftime("%Y-%m-%d"))

#Get dates intersection  between SPADS and ML models
 
idx_intsct= mat_spads.index.intersection(mat_in_norm[ plot_date :mat_in_norm.index[-1]].index)
#Slice SPADS Matrix with the intersect dates      
mat_spads= mat_spads[idx_intsct [0]: idx_intsct[-1]]

mshorefor= mat_spads.ShoreFor.values.astype('float32')
mshorefor = pd.Series(mshorefor,index=idx_intsct)

mspads= mat_spads.SPADS.values.astype('float32')
mspads = pd.Series(mspads,index=idx_intsct)


# mspads.to_csv(rootdir+'Codes_forsubmission/Data/mspads.csv')
# mshorefor.to_csv(rootdir+'Codes_forsubmission/Data/mshorefor.csv')
mat_in.to_csv(rootdir+'Codes_forsubmission/Data/inputs_target.csv')









