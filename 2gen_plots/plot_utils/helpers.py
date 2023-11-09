#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:22:40 2023

@author: egom802
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from scipy.stats import pearsonr

from plot_utils.Taylor_class import TaylorDiagram

def taylor_plot(mat_hyb,mat_cnn,inputs,cvd_fmap):
    data = np.squeeze(inputs.yout)   
    refstd = np.std(data)    # Reference standard deviation of observations
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
    ##########################CNN Taylor Diagram###############################
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
    ##########################CNN-LSTM Taylor Diagram##########################
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
    ###########################################################################
    # Add figure legend
    fig.legend(dia.samplePoints,
                [ p.get_label() for p in dia.samplePoints ],
                numpoints=1, prop=dict(size=12), loc='upper right')
    #Add subplots titles
    fig.text(0.25, 0.87, 'CNN', ha='center',fontsize=fs)
    fig.text(0.75, 0.87, 'CNN-LSTM', ha='center',fontsize=fs)
    #Uncomment to save plot
    plt.savefig('./figures/Fig8.png',
                bbox_inches='tight',dpi=300)

def ts_plot(mat_hyb,mat_cnn,inputs,cvd_fmap):
    """   """
    #Envelopes
    lower= mat_cnn.min(axis=1)
    upper= mat_cnn.max(axis=1)
    mean= mat_cnn.mean(axis=1)
    lower_hyb= mat_hyb.min(axis=1)
    upper_hyb= mat_hyb.max(axis=1)
    mean_hyb= mat_hyb.mean(axis=1)
    ######################PLOT FIGURE 7########################################
    fig,ax = plt.subplots(3,1,figsize=(11.8,6.8))
    fs= 20
    fs_tk= 14
    fs_leg= 12
    colors = cvd_fmap(np.linspace(0, 1, 4))
    #Benchmarks
    ax[0].plot(inputs.yout,'.k', markersize=2)
    ax[0].plot(mat_cnn.spads,color=colors[0],label=r'$SPADS$')
    ax[0].plot(mat_cnn.shorefor,color=colors[1],label=r'$ShoreFor$')
    ax[0].xaxis.set_major_locator(mdates.YearLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[0].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
    ax[0].grid()
    ax[0].legend(loc='lower right',fontsize=fs_leg)
    ax[0].tick_params(axis='both', which='major', labelsize=fs_tk)
    #CNN ensemble
    ax[1].plot(inputs.yout,'.k', markersize=2)
    ax[1].fill_between(mat_cnn.index, y1=lower, y2=upper, 
                 alpha=0.3, color=colors[2])
    ax[1].plot(mat_cnn.index, mean, alpha=0.9,
                  color=colors[2], label= "CNN")
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[1].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
    ax[1].grid()
    ax[1].legend(loc='lower right',fontsize=fs_leg)
    ax[1].tick_params(axis='both', which='major', labelsize=fs_tk)
    #CNN-LSTM ensemble
    ax[2].plot(inputs.yout,'.k', markersize=2)
    ax[2].fill_between(mat_hyb.index, y1=lower_hyb, y2=upper_hyb, 
                 alpha=0.3, color=colors[3])
    ax[2].plot(mat_hyb.index, mean_hyb, alpha=0.9,
                  color=colors[3], label= "CNN-LSTM")
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax[2].set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
    ax[2].grid()
    ax[2].legend(loc='lower right',fontsize=fs_leg)
    ax[2].tick_params(axis='both', which='major', labelsize=fs_tk)

    plt.rcParams.update({'font.size': 28})
    fig.text(0.04, 0.5, 'Cross-shore displacement ' r'$[m]$', va='center',
         rotation='vertical',fontsize=fs)
    #Uncomment to save plot
    plt.savefig('./figures/Fig7.png',
                bbox_inches='tight',dpi=300)
    
def heat_plot(mat_hyb,mat_cnn,inputs):
    
    mean_hyb= mat_hyb.mean(axis=1)
    mean_cnn= mat_cnn.mean(axis=1)
    ##########################PLOT QQ Plot ######################################
    plt.style.use('default')
    fig,axs1 = plt.subplots(2,2,figsize=(12,10))

    qqx= inputs.yout

    df= pd.DataFrame(columns=("qqx","yann","yspads","yshorefor"))
    df["qqx"]= qqx; df["yann"]= mean_hyb; df["yspads"]= mat_hyb.spads;
    df["yshorefor"]= mat_hyb.shorefor; df["ycnn"]= mean_cnn
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
    
    #SPADS
    axs1[0,1].plot(df.qqx,df.qqx,'--w')
    axs1[0,1].set_facecolor(color_rgba)
    axs1[0,1].hist2d(df.qqx, df.yspads,bins=bins)
    axs1[0,1].set_xlim([50, 75])
    axs1[0,1].set_ylim([50, 75])
    axs1[0,1].set_title('SPADS',fontsize = fs)

    #ShoreFor
    axs1[1,1].plot(df.qqx,df.qqx,'--w')
    axs1[1,1].set_facecolor(color_rgba)
    im=axs1[1,1].hist2d(df.qqx, df.yshorefor,bins=bins)
    axs1[1,1].set_xlim([50, 75])
    axs1[1,1].set_ylim([50, 75])
    axs1[1,1].set_title('ShoreFor',fontsize = fs)

    cbar= fig.colorbar(im[3],ax=axs1[:, :], shrink=0.6)
    cbar.ax.tick_params(labelsize=fs) 

    plt.rcParams.update({'font.size': 28})
    fig.text(0.45, 0.04, 'Measured cross-shore displacement ' r'$[m]$',
             ha='center',fontsize=fs)
    fig.text(0.04, 0.5, 'Modelled cross-shore displacement ' r'$[m]$',
             va='center', rotation='vertical',fontsize=fs)
    #Uncomment to save plot
    plt.savefig('./figures/Fig9.png',
                bbox_inches='tight',dpi=300)
    
def coverage_plot(mat_hyb,mat_cnn,inputs):
    
    #Create common dataframe and plot
    yresults= pd.DataFrame(mat_hyb.spads,index=mat_hyb.index,columns=["hyb_m"])
    yresults["spads"]=mat_hyb.spads; yresults["shorefor"]=mat_hyb.shorefor
    yresults["hyb_m"]=mat_hyb.mean(axis=1)
    yresults["hyb_u"]=mat_hyb.max(axis=1)
    yresults["hyb_l"]=mat_hyb.min(axis=1)
    yresults["obs"]= inputs.yout
    ######## Count observations within the shaded area, shift +- 1 day#########
    #Set the number of days to shift
    n_shift= 0

    #Observations that are within the original shade
    obs_shade_bool= yresults["obs"].gt(
        yresults["hyb_m"]) & yresults["obs"].lt(yresults["hyb_u"])

    #Shift time series + 1 day
    obs_shift=yresults.shift(n_shift, freq='D')
    obs_shift_bool= yresults["obs"].gt(
        obs_shift["hyb_l"]) & yresults["obs"].lt(obs_shift["hyb_u"])

    #Shift time series - 1 day
    obs_shift_neg=yresults.shift(-n_shift, freq='D')
    obs_shift_bool_neg= yresults["obs"].gt(
        obs_shift_neg["hyb_l"]) & yresults["obs"].lt(obs_shift_neg["hyb_u"])

    #Get dates intersection  between original and extended time series
    idx_intsct= obs_shade_bool.index.intersection(obs_shift_bool.index)
    #Cut dataframe by intersected dates
    obs_shift_bool= obs_shift_bool[idx_intsct]
    obs_shift_bool_neg= obs_shift_bool_neg[idx_intsct]

    #Combine booleans
    bool_comb= np.logical_or(obs_shift_bool_neg.values, obs_shift_bool.values)
    bool_comb= np.logical_or(bool_comb,obs_shade_bool.values)

    #Apply boolean to filter observations that lay +-1 day shade
    or_shift= yresults[bool_comb]

    #Percentage of observations 
    obs_perc=len(or_shift)/len(yresults)*100
    ###########################################################################
    #Plot the original + shifted 
    #For plotting aesthetics
    color= "lightskyblue"; alpha= 1;
    fs_tk= 12; fs_leg= 11; ms= 4
    fs= 20
    fig,ax = plt.subplots(figsize=(11.8,4))

    plt.fill_between(yresults.index, y1=yresults["hyb_l"], y2=yresults["hyb_u"], 
                     alpha=alpha, color=color, label="Min-max envelope")
    plt.plot(yresults["obs"],'.k',ms=ms,label= "Outside envelope " + "("+ str( 100- round(obs_perc,1)) + "%" + ")")
    plt.plot(or_shift["obs"],'.r',ms=ms, label="Inside envelope " + "("+ str( round(obs_perc,1)) + "%" + ")" )
    plt.legend()

    ax.set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
    ax.legend(loc='lower right',fontsize=fs_leg)
    ax.tick_params(axis='both', which='major', labelsize=fs_tk)
    fig.text(0.04, 0.5, 'Cross-shore \ndisplacement ' r'$[m]$', va='center',
      rotation='vertical',fontsize=fs)  
    
    #Legend order
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    order = [2,1,0]

    #add legend to plot
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               loc= "lower right", fontsize=10)

    # Uncomment to save plot
    plt.savefig('./figures/Fig10.png',
                bbox_inches='tight',dpi=300)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
