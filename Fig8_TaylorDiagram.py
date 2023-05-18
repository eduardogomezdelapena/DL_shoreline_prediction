#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 2023

@author: EGP
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

#CVisionDeficiency friendly palette from  Crameri, F. (2018). 
#Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862
cm_data = np.loadtxt("./colormaps/roma.txt")
cvd_fmap = LinearSegmentedColormap.from_list('cvd_friendly', cm_data)

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

#ShoreFor and SPADS are included as the 11th, and 12th column 
#in the mat_hyb and mat_cnn
#####################PLOT TAYLOR DIAGRAMS######################################
from Taylor_diagram_ML_modified import TaylorDiagram

data = np.squeeze(inputs.yout)   
refstd = np.std(data)         # Reference standard deviation of observations

#Labels for plotting
labels = {
  "m1": "DL1",
  "m2":"DL2",    
  "m3":"DL3", 
  "m4":"DL4", 
  "m5":"DL5", 
  "m6":"DL6", 
  "m7":"DL7", 
  "m8":"DL8", 
  "m9":"DL9", 
  "m10":"DL10",   
  "mspads":"SPADS",
  "mshorefor":"ShoreFor"#,   
}

plt_labels=[]
for k in labels.keys():
    plt_labels.append(labels[k])
  
# Compute stddev and correlation coefficient of CNN modelS
samples = np.array([ [np.std(mat_cnn[colname].values),
                      pearsonr(data, mat_cnn[colname].values)[0]]
                          for (index, colname) in enumerate(mat_cnn)])        
dl_samples=samples[0:-2] #Last 2 samples are spads and shorefor time series

#For tuning the plot (aesthetic purposes)
ms=15
fs= 20
fs_cont=16
colors = cvd_fmap(np.linspace(0, 1, len(samples)+4))
subcolors = colors[6:]

fig = plt.figure(figsize=(13.9, 6.4))
##########################CNN Taylor Diagram###################################
dia = TaylorDiagram(refstd, fig=fig,  rect=121,  label="Reference",
                        srange=(0.3, 1.5))
# Add grid
for i, (stddev, corrcoef) in enumerate(dl_samples):
    dia.add_sample(stddev, corrcoef,
                    marker='P', ms=ms, ls='',
                    mfc=subcolors[i], mec=subcolors[i],
                    label=plt_labels[i])
#Add SPADS marker
dia.add_sample(samples[-1][0], samples[-1][1],
                    marker='o', ms=ms, ls='', 
                    mfc=colors[4],  mec=colors[4],
                    label=plt_labels[-1])

#Add Shorefor marker
dia.add_sample(samples[-2][0], samples[-2][1],
                    marker="^", ms=ms, ls='',
                    mfc=colors[0], mec=colors[0],                    
                    label=plt_labels[-2])

# Add RMS contours, and label them
contours = dia.add_contours(colors='0.5')
plt.clabel(contours, inline=1, fontsize=fs_cont, fmt='%.2f')
dia.add_grid()
##########################CNN-LSTM Taylor Diagram##############################
dia2 = TaylorDiagram(refstd, fig=fig,  rect=122,  label="Reference",
                        srange=(0.3, 1.5))

# Compute stddev and correlation coefficient of CNN-LSTM models
samples = np.array([ [np.std(mat_hyb[colname].values), 
                      pearsonr(data, mat_hyb[colname].values)[0]]
                          for (index, colname) in enumerate(mat_hyb)])  
dl_samples=samples[0:-2]

for i, (stddev, corrcoef) in enumerate(dl_samples):
    dia2.add_sample(stddev, corrcoef,
                    marker='P', ms=ms, ls='',
                    mfc=subcolors[i], mec=subcolors[i],
                    label=plt_labels[i])

#Add SPADS marker
dia2.add_sample(samples[-1][0], samples[-1][1],
                    marker='o', ms=ms, ls='', 
                    mfc=colors[4],  mec=colors[4],
                    label=plt_labels[-1])

#Add Shorefor marker
dia2.add_sample(samples[-2][0], samples[-2][1],
                    marker="^", ms=ms, ls='',
                    mfc=colors[0], mec=colors[0],                    
                    label=plt_labels[-2])

# Add RMS contours, and label them
contours = dia2.add_contours(colors='0.5')
plt.clabel(contours, inline=1, fontsize=fs_cont, fmt='%.2f')
dia2.add_grid()
###############################################################################
# Add figure legend
fig.legend(dia.samplePoints,
                [ p.get_label() for p in dia.samplePoints ],
                numpoints=1, prop=dict(size=12), loc='upper right')
#Add subplots titles
fig.text(0.25, 0.87, 'CNN ', ha='center',fontsize=fs)
fig.text(0.75, 0.87, 'CNN-LSTM', ha='center',fontsize=fs)
#Uncomment to save plot
# plt.savefig('./Figures/Fig8.png',
#             bbox_inches='tight',dpi=300)