#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:34:14 2023

@author: egom802
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np

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
        seq_x, seq_y = sequences[i:end_ix,
                                 :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
     return np.array(X), np.array(y)
 
