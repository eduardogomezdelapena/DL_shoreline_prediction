#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created  May 2023

@author: EGP
"""

import os
import random
import numpy as np
import pandas as pd
import math
import scipy.stats

from sklearn import preprocessing

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from datetime import  timedelta
from sklearn.metrics import mean_squared_error
#%%
#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

#%%############################################################################
#Import csv files, convert to datetime 

mat_pc= pd.read_csv('./Data/Originals/matrix_time_first100PC.csv')

PC_number= 10
key_list= ['PC('+str(x)+')' for x in range(1,PC_number+1)] #Create PC strings
key_list= ["Y","M","D","h","m","s"] + key_list #Append time columns
mat_pc= mat_pc[key_list] #Obtain desireed PCs + time columns

mat_waves=  pd.read_csv('./Data/Originals/matrix_time_Hs_Tp_Dir.csv')
mat_out= pd.read_csv('./Data/Originals/matrix_time_outputy.csv')


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
#%%################################################################################
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
#%%################################################################################
######################CONCATENATE INPUTS#######################################
#Here we choose which inputs go to the NN
# sum_bool= sum([mat_pc, mat_waves,ar])

#inputs = [mat_pc, mat_waves, mat_out]
inputs = [mat_waves, mat_out]
         
#%%#######################HOMOGENEOUS SERIES#######################################
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
#%%###############################################################################
#Import SPADS, Shorefor csv files, convert to datetime 

mat_spads= pd.read_csv('./Data/Originals/matrix_time_spads_shorefor_obs.csv') 

#Convert date columns to datetime 
mat_spads= columns_to_datetime(mat_spads)

#The forecast for Tairua is from July 2014 previous data is training in SPADS and
#ShoreFor

date_forecast= '2014-07-01'

mat_spads = mat_spads[date_forecast:mat_spads.index[-1]]

#Make sure all time -series end at the same date
#Get dates intersection with mat_spads
idx_intsct= mat_spads.index.intersection(mat_in.index)

#Slice Matrix inputs with the intersect dates      
mat_in= mat_in[mat_in.index[0]: idx_intsct[-1]]
#Slice Matrix outputs with the intersect dates                
mat_out= mat_out[mat_out.index[0]: idx_intsct[-1]]   

#%%##################TRAIN, DEV AND TEST SPLITS#################################

#Manually split by date 
#Remember: Tairua the forecast is from July 2014 (2014-07-01), previous data 
#is training in SPADS and ShoreFor

#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))

#%%##########################DATA NORMALIZATION################################
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

_, mat_in = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)


#%%##################TRAIN, DEV AND TEST SPLITS################################

#Manually split by date 
#Remember: Tairua the forecast is from July 2014 (2014-07-01) on
#previous data is training in SPADS and ShoreFor

train = mat_in[mat_in.index[0]:train_date].values.astype('float32')

#Development set (2 years before the test set)
devinit_date=pd.to_datetime(train_date) + timedelta(days=1)
devinit_date= str(devinit_date.strftime("%Y-%m-%d"))
dev_date=pd.to_datetime(date_forecast) - timedelta(days=1)
dev_date= str(dev_date.strftime("%Y-%m-%d"))
dev= mat_in[devinit_date:dev_date].values.astype('float32')

#Test set, depends on study site
test = mat_in[date_forecast:mat_in.index[-1]].values.astype('float32')
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
 
idx_intsct= mat_spads.index.intersection(mat_in[ plot_date :mat_in.index[-1]].index)
#Slice SPADS Matrix with the intersect dates      
mat_spads= mat_spads[idx_intsct [0]: idx_intsct[-1]]

mshorefor= mat_spads.ShoreFor.values.astype('float32')
mspads= mat_spads.SPADS.values.astype('float32')

#%% Error definitions
def index_mielke(s, o):
    """
	index of agreement
	Modified Mielke (2015 Duveiller) 
	input:
        s: simulated
        o: observed
    output:
        im: index of mielke
    """
    d1= np.sum((o-s)**2)
    d2= np.var(o)+np.var(s)+ ( np.mean(o)-np.mean(s)  )**2
    im= 1 - (((len(o)**-1)*d1)/d2)
    return im

#%%###################Define Loss functions: NEURAL NETWORK##################
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1- K.square(r)

def mielke_loss(y_true, y_pred):
    """ Mielke index 
    if pearson coefficient (r) is zero or positive use kappa=0
    otherwise see Duveiller et al. 2015
    """
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    
    diff= math_ops.squared_difference(y_true, y_pred) 
    d1= K.sum(diff)
    d2= K.var(y_true)+K.var(y_pred)+ tf.math.square((K.mean(y_true)-K.mean(y_pred)))
   
    if correlation_coefficient_loss(y_true, y_pred) < 0:
        kappa = tf.multiply(K.abs( K.sum(tf.multiply(xm,ym))),2)
        loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / (d2 +kappa))
    else:
        loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / d2 )
    
    return 1- loss
#%%CNN model
loss= mielke_loss
min_delta= 0.001

def cnn_custom(train_x, train_y, dev_x, dev_y, cfg):
    print("--------------------------------")
    print("Model:", cfg)
    set_seed(33)
    # define model    # create configs
    n_filters, n_kernels, n_drop, n_epochs,n_batch = cfg    
    
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels, activation='relu',
                     input_shape=(n_steps_in, n_features)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels, activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten()) 
    model.add(Dense(100, activation='relu')) 
    model.add(Dropout(n_drop))       
    model.add(Dense(1))    
    model.compile(optimizer='adam', loss=loss)
    # fit model
    es = EarlyStopping(patience=10, verbose=2, min_delta=min_delta,
                       monitor='val_loss', mode='auto', restore_best_weights=True)

    history= model.fit(train_x, train_y, validation_data=(dev_x, dev_y),
                         batch_size=n_batch, epochs=n_epochs, verbose=2, callbacks=[es])  
 
    return model, history
#%%############################################################################
# Load Grid Search Hyperparameters
scores = list()   

cfg_list=pd.read_csv('./Data/10best_hyp_Wavesonly_Mielke_CNN.csv')
cfg_list= cfg_list[["f","k","D","e","b"]]

cfg_list= cfg_list.values.tolist()

for i in range(len(cfg_list)):     
    for element in range(len(cfg_list[i])):
        if element != 2:
            cfg_list[i][element] = int(cfg_list[i][element])   
#%%#Run model configurations in loop###########################################
#Predefine empty dataframe
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])

#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)

for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index]) 
    testdl = model.predict(test_x)     

    yresults.iloc[:,index]= scaler.inverse_transform(testdl)
    
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))
#%% EXPORT ENSEMBLE

#Add spads and shorefor
yresults["spads"]=mspads; yresults["shorefor"]=mshorefor

#Uncomment to save 
# yresults.to_csv('./Data/CNN_ensemble2.csv')

#Metrics 
# rmse_arr=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
# pear_arr=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
# mielke_arr=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])