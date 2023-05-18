#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: EGP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

mshorefor= mat_hyb.shorefor
mspads= mat_hyb.spads

mean_hyb= mat_hyb.mean(axis=1)
mean_cnn= mat_cnn.mean(axis=1)
##########################PLOT QQ Plot ######################################
plt.style.use('default')
fig,axs1 = plt.subplots(2,2,figsize=(12,10))

qqx= inputs.yout
x = np.sort(qqx,axis=0)

df= pd.DataFrame(columns=("qqx","yann","yspads","yshorefor"))
df["qqx"]= qqx; df["yann"]= mean_hyb; df["yspads"]= mspads;
df["yshorefor"]= mshorefor; df["ycnn"]= mean_cnn
df=df.sort_values("qqx")

bins= 20
fs= 20
# colors = cvd_fmap(np.linspace(0, 1, 20))
# color_rgba= colors[0]
color_rgba= "#440154ff"

#CNN
axs1[0,0].plot(df.qqx,df.qqx,'--w')
axs1[0,0].hist2d(df.qqx, df.ycnn,bins=bins)
axs1[0,0].set_facecolor(color_rgba)
axs1[0,0].set_xlim([50, 75])
axs1[0,0].set_ylim([50, 75])
axs1[0,0].set_title('CNN',fontsize = fs)

#Hybrid
axs1[1,0].plot(df.qqx,df.qqx,'--w')
axs1[1,0].hist2d(df.qqx, df.yann,bins=bins)
axs1[1,0].set_facecolor(color_rgba)
axs1[1,0].set_xlim([50, 75])
axs1[1,0].set_ylim([50, 75])
axs1[1,0].set_title('CNN-LSTM',fontsize = fs)

#Secon graph
axs1[0,1].plot(df.qqx,df.qqx,'--w')
axs1[0,1].set_facecolor(color_rgba)
axs1[0,1].hist2d(df.qqx, df.yspads,bins=bins)
axs1[0,1].set_xlim([50, 75])
axs1[0,1].set_ylim([50, 75])
axs1[0,1].set_title('SPADS',fontsize = fs)

#Secon graph
axs1[1,1].plot(df.qqx,df.qqx,'--w')
axs1[1,1].set_facecolor(color_rgba)
im=axs1[1,1].hist2d(df.qqx, df.yshorefor,bins=bins)
axs1[1,1].set_xlim([50, 75])
axs1[1,1].set_ylim([50, 75])
axs1[1,1].set_title('ShoreFor',fontsize = fs)

cbar= fig.colorbar(im[3],ax=axs1[:, :], shrink=0.6)
cbar.ax.tick_params(labelsize=fs) 

plt.rcParams.update({'font.size': 28})
fig.text(0.45, 0.04, 'Measured cross-shore displacement ' r'$[m]$', ha='center',fontsize=fs)
fig.text(0.04, 0.5, 'Modelled cross-shore displacement ' r'$[m]$', va='center', rotation='vertical',fontsize=fs)
#Uncomment to save plot
# plt.savefig('./Figures/Fig9.png',
#             bbox_inches='tight',dpi=300)
