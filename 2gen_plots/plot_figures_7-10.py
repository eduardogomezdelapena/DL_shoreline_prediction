#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created May 2023

@author: EGP
"""
from plot_utils.helpers import taylor_plot, ts_plot, heat_plot, coverage_plot
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import os

#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

#CVisionDeficiency friendly palette from  Crameri, F. (2018). 
#Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862
cm_data = np.loadtxt("./colormaps/roma.txt")
cvd_fmap = LinearSegmentedColormap.from_list('cvd_friendly', cm_data)

#Import Hybrid ensemble time series (test period)
mat_hyb= pd.read_csv('../1run_models/output/Hybrid_ensemble.csv')  
mat_hyb['Datetime'] = pd.to_datetime(mat_hyb['Datetime'])
mat_hyb = mat_hyb.set_index(['Datetime'])

#Import CNN ensemble time series (test period)
mat_cnn= pd.read_csv('../1run_models/output/CNN_ensemble.csv')  
mat_cnn['Datetime'] = pd.to_datetime(mat_cnn['Datetime'])
mat_cnn = mat_cnn.set_index(['Datetime'])

#Import Observed Shoreline
inputs= pd.read_csv('../data/inputs_target.csv')  
inputs['Datetime'] = pd.to_datetime(inputs['Datetime'])
inputs = inputs.set_index(['Datetime'])

#Get dates intersection  between  ML ensemble and inputs
idx_intsct= inputs.index.intersection(mat_hyb.index)
inputs= inputs[idx_intsct [0]: idx_intsct[-1]]

###########################PLOTS###############################################
taylor_plot(mat_hyb,mat_cnn,inputs,cvd_fmap)

ts_plot(mat_hyb,mat_cnn,inputs,cvd_fmap)

heat_plot(mat_hyb,mat_cnn,inputs)

coverage_plot(mat_hyb,mat_cnn,inputs)