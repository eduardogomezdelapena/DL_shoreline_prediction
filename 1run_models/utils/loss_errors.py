#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: EGP
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
import numpy as np
#%% Error definitions
def index_mielke(s, o):
    """
	index of agreement
	Modified Mielke (Duveiller 2015) 
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
    d2= K.var(y_true)+K.var(y_pred)+ tf.math.square(
        (K.mean(y_true)-K.mean(y_pred)))
   
    if correlation_coefficient_loss(y_true, y_pred) < 0:
        kappa = tf.multiply(K.abs( K.sum(tf.multiply(xm,ym))),2)
        loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / (d2 +kappa))
    else:
        loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / d2 )
    return 1- loss