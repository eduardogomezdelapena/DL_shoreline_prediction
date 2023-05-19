#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:03:35 2021

@author: eduardo
"""

from os.path import join as pjoin
import scipy.io as sio
import matplotlib.pyplot as plt 
import os
import csv
from datetime import date

from datetime import timedelta
import numpy as np
import pandas as pd

gendir='/home/eduardo/Documents/PhD_UoA/ML_paper/Tairua/'


data_dir = (gendir + 'Original_PC_wave_shoreline_csv')
mat_fname = pjoin(data_dir, 'shoreline_Tairua_v2_JM.mat')


mat_contents = sio.loadmat(mat_fname)
sorted(mat_contents.keys())


data= mat_contents['Data']

plt.plot(data)
matlab_datenum= mat_contents['time']
matlab_datenum= [ int(i) for i in matlab_datenum ]


py_dates=[]

for i in range(len(matlab_datenum)):
    python_datetime = date.fromordinal(int(matlab_datenum[i])) + timedelta(days=matlab_datenum[i]%1) - timedelta(days = 366)
    py_dates.append(python_datetime)


y= pd.DataFrame(data, index= pd.to_datetime(py_dates), columns= ["data"])

os.chdir(data_dir)


#ONLY 3 DECIMALS
y= y.round(3)
y['h']= 0; y['m']= 0; y['s']= 0


y = y[["h","m","s","data"]]

y.to_csv('file_prueba.csv', header=None )

#Change .csv file , replace "-" for ","
text = open("file_prueba.csv", "r")
text = ''.join([i for i in text]) \
    .replace("-", ",")
x = open("file_prueba.csv","w")
x.writelines(text)
x.close()


#Add headers "Y,M,D,h,m,s"
df = pd.read_csv("file_prueba.csv",header=None)
df= df.round(3)
df.to_csv("file_prueba.csv", header=["Y","M","D","h","m","s","yout"],index=False)


plt.plot(y,'orangered',label=r'$Shoreline$')






























