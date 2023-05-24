#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May  2023
@author: EGP
"""
import os
import numpy as np
import pandas as pd
import math
#%%#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)
#%%############################################################################
#Import csv files, convert to datetime 
mat_pc= pd.read_csv('./Originals/matrix_time_first100PC.csv')  
PC_number= 10

key_list= ['PC('+str(x)+')' for x in range(1,PC_number+1)] #Create PC strings
key_list= ["Y","M","D","h","m","s"] + key_list #Append time columns
mat_pc= mat_pc[key_list] #Obtain desireed PCs + time columns

mat_waves=  pd.read_csv('./Originals/matrix_time_Hs_Tp_Dir.csv')
mat_out= pd.read_csv('./Originals/matrix_time_outputy.csv')

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
#%%########################Shift###############################################
mat_out=mat_out.shift(periods=-3) 
#SMoothed time series  
mat_out=mat_out.rolling(7).mean()  
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
#%%###################CONCATENATE INPUTS#######################################
#Here we choose which inputs go to the NN
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

mat_spads= pd.read_csv('./Originals/matrix_time_spads_shorefor_obs.csv') 

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

##Save
# mat_in.to_csv('./inputs_target.csv')
# mat_spads.SPADS.to_csv('./mspads.csv')
# mat_spads.ShoreFor.to_csv('./mshorefor.csv')