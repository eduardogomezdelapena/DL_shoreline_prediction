#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created May 2023

@author: EGP
"""
from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import pandas as pd
import math
import scipy.stats
from datetime import  timedelta
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Flatten, RepeatVector,TimeDistributed
from keras.layers.convolutional import Conv1D,MaxPooling1D
#%%#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)
#%%#Import Observed Shoreline positions
mat_in= pd.read_csv('../data/inputs_target.csv')  
mat_in['Datetime'] = pd.to_datetime(mat_in['Datetime'])
mat_in = mat_in.set_index(['Datetime'])
mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)
# Import SPADS 
mspads= pd.read_csv('../data/mspads.csv')
mspads['Datetime'] = pd.to_datetime(mspads['Datetime'])
mspads = mspads.set_index(['Datetime'])
#Import ShoreFor
mshorefor= pd.read_csv('../data/mshorefor.csv')
mshorefor['Datetime'] = pd.to_datetime(mshorefor['Datetime'])
mshorefor = mshorefor.set_index(['Datetime'])
#%%######################DATA NORMALIZATION####################################
#Remember: Tairua the forecast is from July 2014 (2014-07-01), previous data 
#is training in SPADS and ShoreFor
date_forecast= '2014-07-01'
#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))
_, mat_in = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)
#%%##################TRAIN, DEV AND TEST SPLITS################################
#Manually split by date 
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
#how many time steps is the network allowed to look back
n_steps_in, n_steps_out =40,1
train_x, train_y = split_sequences(train, n_steps_in, n_steps_out)
dev_x, dev_y = split_sequences(dev, n_steps_in, n_steps_out)
test_x, test_y = split_sequences(test, n_steps_in, n_steps_out)
# # the dataset knows the number of features, e.g. 2
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]
#%%###################Fix random seed##########################################
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
#%%####################CONV-LSTM Network#######################################
loss= mielke_loss
min_delta= 0.001
def cnn_custom(train_x, train_y, dev_x, dev_y, cfg):
    print("--------------------------------")
    print("Model:", cfg)
    set_seed(33)
    # define model    # create configs
    n_filters, n_kernels, n_mem,n_drop,n_epochs,n_batch = cfg    
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(n_mem, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(Dropout(n_drop))      
    model.add(TimeDistributed(Dense(1)))    
    model.compile(optimizer='adam', loss=loss)
    # fit model
    es = EarlyStopping(patience=10, verbose=2, min_delta=min_delta, 
                       monitor='val_loss', mode='auto',
                       restore_best_weights=True)
    history= model.fit(train_x, train_y, validation_data=(dev_x, dev_y),
                         batch_size=n_batch, epochs=n_epochs, verbose=2,
                         callbacks=[es])  
    return model, history
#%%###################### Load Grid Search Hyperparameters#####################
scores = list()   
cfg_list=pd.read_csv('./hyp/10best_hyp_Wavesonly_Mielke_Hybrid.csv')
cfg_list= cfg_list[["f","k","m","D","e","b"]]
cfg_list= cfg_list.values.tolist()
for i in range(len(cfg_list)):     
    for element in range(len(cfg_list[i])):
        #Position where dropout percentage is
        if element != 3:
            cfg_list[i][element] = int(cfg_list[i][element])
#%%#Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in-1)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])
#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
#train the models
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index]) 
    testdl = model.predict(test_x)     

    yresults.iloc[:,index]= scaler.inverse_transform(testdl.reshape(
        testdl.shape[0]*testdl.shape[1],testdl.shape[2]))
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))
#%% EXPORT ENSEMBLE
#Cut spads and shorefor to match the DL test time series output
yresults["spads"]=mspads[plot_date :mspads.index[-1]]
yresults["shorefor"]=mshorefor[plot_date :mshorefor.index[-1]]
#Uncomment to save 
# yresults.to_csv('./output/Hybrid_ensemble.csv')

#Metrics 
# rmse_arr=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
# pear_arr=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
# mielke_arr=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])