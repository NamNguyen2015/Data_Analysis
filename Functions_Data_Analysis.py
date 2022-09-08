#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:00:28 2022

@author: namnguyen
"""

import pandas as pd
import numpy as np
import re
import os
import pathlib
import matplotlib.pyplot as plt
#import seaborn as sns
from statistics import mean
import matplotlib.cbook as cbook
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# %% get path file Design_limits and Angles
path = pathlib.Path(os.getcwd())
path_input = path/'Input Files'

cr_path = path/'Input Files/Design/'
df_design_limits = pd.read_excel(
    cr_path/'Design_Limits &Angles.xlsx', sheet_name=None, header=0, index_col=None)

for k, v in df_design_limits.items():
    v.drop(v.columns[0], axis=1, inplace=True)
    v.set_index(v.columns[0], inplace=True)

# %%


def sayHello():
    print("Hello world!")
 
#%% This funtion is apply to remove the strange dispersion in P-17-2&4
import copy
def dispersion(y,eps):
    index_list=find_positions(y, eps)
    offset_list=find_offset(y,eps)
    N=len(index_list)
    #ynew=copy.deepcopy(y)
    for i in range(len(y)):
        for k in range(N):
            if i>index_list[k]:
                y[i]=y[i]-offset_list[k]
    #ynew=y
    return y
            
    
#%%%Pz_i_j

def Pz_i_j(df,i,j):
    try:
            
            
        df_P_i_j=df[(df['Pier']=='P')&(df['No']==i)&(df['Point']==j)]
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        #fig, ax =plt.subplots(constrained_layout=False)
        x=df_P_i_j["Date"]
        y=df_P_i_j["Z_mm"]
        ax.plot(x,y,  color='b', linestyle="-", label="Z")
        ax.set_xlabel("Date")
        ax.set_ylabel("mm")
        ax.set_title("Pz-"+str(i)+"-"+str(j))
        ax.legend()
        fig.autofmt_xdate()
      
    except:
        fig = None
        pass
    return fig

#%%Pxy_i_j

def Pxy_i_j(df,i,j):
    
    try:
        
        df_P_i_j=df[(df['Pier']=='P')&(df['No']==i)&(df['Point']==j)]
        #fig,ax=plt.subplots(constrained_layout=False)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        x=df_P_i_j["Date"]
        
        y1=df_P_i_j["X_mm"]
        y2=df_P_i_j["Y_mm"]
        ax.plot(x,y1, label="X",color="blue")
        ax.plot(x,y2, label="Y", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("mm")
        ax.set_title("Pxy-"+str(i)+"-"+str(j))
        ax.legend()
        fig.autofmt_xdate()
    except:
        fig = None
        pass
    return fig
        

# %% Function to plot Pz in process


def Pz_Plot(df, i):
    try:
        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        f = pd.pivot_table(
            df_P_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        #x_cr, y_cr = correction(epsilon_pz[i-1], x, y)

        #fig, ax = plt.subplots(constrained_layout=False)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x,y, color='r', label='Z')
       # ax.plot(x_cr, y_cr, color='g', label='Z-correction')
        #ax.hlines(y=settlement_pz[i-1], color='b', xmin=x_cr[0], xmax=x_cr[len(
         #   x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('Pz-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass

    return fig


# %%Function to plot Dz in process
# =============================================================================
# def Dz_Plot(df, i):
#     try:
#         df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
#         f = pd.pivot_table(
#             df_D_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
#         x = f.reset_index()["Date"]
#         y = f.reset_index()["Z_mm"]
#         #x_cr, y_cr = correction(epsilon_dz[i-1], x, y)
# 
#         fig, ax = plt.subplots(constrained_layout=True)
#         ax.plot(x,y, color='r', label='Z',linestyle='-')
# 
#         #ax.plot(x_cr, y_cr, color='g', label='Z-correction')
# 
#         #ax.hlines(y=settlement_dz[i-1], color='b', xmin=x[0], xmax=x[len(
#          #   x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
#         ax.set_title('Dz-'+str(i))
#         ax.set_xlabel('Date')
#         ax.set_ylabel('mm')
#         ax.legend(loc='best')
#         fig.autofmt_xdate()
#     except:
#         fig=None
#         pass
# # =============================================================================
# #     df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
# #     fig = sns.relplot(data=df_D_i, x='Date', y='Z_mm', estimator=np.mean, col="Pier",
# #                       row="No", ci=None, facet_kws={'sharey': False, 'sharex': True}, kind='line')
# #     
# #     #fig.autofmt_xdate()
# # =============================================================================
#     return fig
# =============================================================================

# %%Dz
def Dz_Plot(df, i):
    epsilon_dz = df_design_limits['D']['EPSILON Z']
    settlement_dz = df_design_limits['D']['SETTLEMENT']
    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        f = pd.pivot_table(
            df_D_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        #x_cr, y_cr = correction(epsilon_dz[i-1], x, y)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x,y, color='r', label='Z',linestyle='-')

        #ax.plot(x_cr, y_cr, color='g', label='Z-correction')

        #ax.hlines(y=settlement_dz[i-1], color='b', xmin=x[0], xmax=x[len(
        #    x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('Dz_-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend()
        fig.autofmt_xdate()
    except:
        fig = None
        pass

    return fig





# %%Function to plot PXY in process
def P_XY_Plot(df, i):
    alphaP = np.deg2rad(df_design_limits['P']['ANGLE'])
    long_Pos = df_design_limits['P']['TOP LONGITUDINAL POSITIVE']
    long_Neg = df_design_limits['P']['TOP LONGITUDINAL NEGATIVE']
    tran_Pos = df_design_limits['P']['TOP TRANSVERSAL POSITIVE']
    tran_Neg = df_design_limits['P']['TOP TRANSVERSAL NEGATIVE']
    try:
        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        df_P_i = df_P_i.assign(
            LONGITUDINAL=df_P_i['X_mm']*np.cos(alphaP[i-1])+df_P_i['Y_mm']*np.sin(alphaP[i-1]))
        df_P_i = df_P_i.assign(
            TRANVERSAL=-df_P_i['X_mm']*np.sin(alphaP[i-1])+df_P_i['Y_mm']*np.cos(alphaP[i-1]))
        f = pd.pivot_table(df_P_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)
        # dfm = df_P_i.melt(['Pier','No','Point','Date','X_mm','Y_mm', 'Z_mm'], var_name='Longitudinal and Traversal',value_name='values')#)
        #fig=sns.relplot(data=dfm, x="Date", y='values', hue='Longitudinal and Traversal',kind='line',estimator=np.mean,col="Pier", row="No",ci=None,facet_kws={'sharey': False, 'sharex': True})

        # fig.data = dfm  # Hack needed to work around bug on v0.11, fixed in v0.12.dev
        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        #fig, ax = plt.subplots(constrained_layout=False)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x, y1, color='r', label='Longitudinal', linestyle='-')
        ax.plot(x, y2, color='g', label='Traversal', linestyle='-')
        ax.hlines(y=long_Pos[i-1], color='b', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Positive')
        ax.hlines(y=long_Neg[i-1], color='g', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Negative')
        ax.hlines(y=tran_Pos[i-1], color='r', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Positive')
        ax.hlines(y=tran_Neg[i-1], color='orange', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Negative')
        ax.set_title('P_XY-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass
    return fig

# %% Function to plot D_XY in process


def D_XY_Plot(df, i):
    alpha = np.deg2rad(df_design_limits['D']['ANGLE'])
    
    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        df_D_i = df_D_i.assign(
            LONGITUDINAL=df_D_i['X_mm']*np.cos(alpha[i-1])+df_D_i['Y_mm']*np.sin(alpha[i-1]))
        df_D_i = df_D_i.assign(
            TRANVERSAL=-df_D_i['X_mm']*np.sin(alpha[i-1])+df_D_i['Y_mm']*np.cos(alpha[i-1]))
        f = pd.pivot_table(df_D_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)
      

        #dfm = df_D_i.melt(['Pier', 'No', 'Point', 'Date', 'X_mm',
         #                 'Y_mm', 'Z_mm'], var_name='col_xy', value_name='val_xy')
        #fig = sns.relplot(data=dfm, x="Date", y='val_xy', hue='col_xy', kind='line', estimator=np.mean,
         #                 col="Pier", row="No", ci=None, facet_kws={'sharey': False, 'sharex': True})
        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        #fig, ax = plt.subplots(constrained_layout=False)
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x,y1, color='r', label='Longitudinal',linestyle='-')
        ax.plot(x,y2, color='b', label='Traversal',linestyle='-')
        
        ax.set_title('D_XY-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        fig=None
        pass
    return fig


# %% Find out unexpected positions where exist peaks or steps

def find_positions(y, epsilon_h):
    # x1=np.diff(x)
    y1 = np.diff(y)
    index_list = list()
    for i in range(len(y1)):
        if abs(y1[i]) > epsilon_h:
            index_list.append(i)
        # elif abs(y1[i]/x1[i])>epsilon_v:
           # index_list.append(i)
        elif y[i] == None:
            index_list.append(i)
            
    return index_list
#%%
def find_offset(y,eps):
    offset_list=list()
    index_list=find_positions(y, eps)
    for k in index_list:
        d=y[k+1]-y[k]
        offset_list.append(d)
    return offset_list

        
        
    
    


# %%a function to find the correction
def correction(epsilon, x, y):
    index_list = find_positions(y, epsilon)

    N = len(index_list)
    # Create 2 array of all segments of dates and values
    # assume that  we alreaddy had a list of y and an index_list inside y

    if N >= 1:
        # This list will contain all the segments of values
        S = [0 for i in range(N+1)]
        # This list will contain all the segments ocorresponding dates
        S_date = [0 for i in range(N+1)]

        S_date[0] = x[:index_list[0]]
        S_date[N] = x[index_list[N-1]+1:]
        S[0] = y[:index_list[0]]  # the first sequence element in the list
        S[N] = y[index_list[N-1]+1:]  # the last sequence element in the list

        for k in range(1, N):
            S_date[k] = x[index_list[k-1]+1:index_list[k]]
            S[k] = y[index_list[k-1]+1:index_list[k]]

        S1_date = S_date.copy()
        # the segments without elements mean that exist a peak between them.
        S1 = S.copy()
        S1_date = [s for s in S_date if s.any(0)]
        # the segments without elements mean that exist a peak between them.
        S1 = [s for s in S if s.any(0)]
        #                                #We dont care about the peaks and these points will be deleted
        #

    # get the average value of each segment
        S2 = S1.copy()
        #Smean=[0 for i in range(1,len(S2)+1)]
        for i in range(1, len(S2)):
           # delta_i=np.mean(S2[i])-np.mean(S2[i-1])
            delta_i = np.mean(S2[i][:2])-np.mean(S2[i-1][-2:])
            S2[i] = S2[i]-delta_i

            x_correct = []
            y_correct = []
            x_correct = np.concatenate(S1_date, axis=0)
            y_correct = np.concatenate(S2, axis=0)

    elif N == 0:
        x_correct = x
        y_correct = y

    return [x_correct, y_correct]


# %%Pzcor_Plot Funtion to plot after correction

def PzCor_Plot(df, i):
    epsilon_pz = df_design_limits['P']['EPSILON Z']
    settlement_pz = df_design_limits['P']['SETTLEMENT']
    try:
        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        f = pd.pivot_table(
            df_P_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        x_cr, y_cr = correction(epsilon_pz[i-1], x, y)

        #fig, ax = plt.subplots(constrained_layout=False)
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        #ax.plot(x,y, color='r', label='Z',linestyle=':')
        ax.plot(x_cr, y_cr, color='g', label='Z-correction')
        ax.hlines(y=settlement_pz[i-1], color='b', xmin=x_cr[0], xmax=x_cr[len(
            x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('P_cor-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass

    return fig


# %%Dzcor_Plot Funtion to plot after correction
def Dzcor_Plot(df, i):
    epsilon_dz = df_design_limits['D']['EPSILON Z']
    settlement_dz = df_design_limits['D']['SETTLEMENT']
    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        f = pd.pivot_table(
            df_D_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        x_cr, y_cr = correction(epsilon_dz[i-1], x, y)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        
        #ax.plot(x,y, color='r', label='Z',linestyle='-')

        ax.plot(x_cr, y_cr, color='g', label='Z-correction')

        ax.hlines(y=settlement_dz[i-1], color='b', xmin=x[0], xmax=x[len(
            x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('D_cor-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        fig = None
        pass

    return fig

# %%P_XYcor_Plot Funtion to plot after correction


def P_XYcor_Plot(df, i):

    alphaP = np.deg2rad(df_design_limits['P']['ANGLE'])
    epsilon_px = df_design_limits['P']['EPSILON X']
    epsilon_py = df_design_limits['P']['EPSILON Y']

    # calculate epsilon for Longitudinal and Tranversal
    eps_L = epsilon_px*np.cos(alphaP)+epsilon_py*np.sin(alphaP)
    eps_T = -epsilon_px*np.sin(alphaP)+epsilon_py*np.cos(alphaP)

    long_Pos = df_design_limits['P']['TOP LONGITUDINAL POSITIVE']
    long_Neg = df_design_limits['P']['TOP LONGITUDINAL NEGATIVE']
    tran_Pos = df_design_limits['P']['TOP TRANSVERSAL POSITIVE']
    tran_Neg = df_design_limits['P']['TOP TRANSVERSAL NEGATIVE']

    try:

        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        df_P_i = df_P_i.assign(
            LONGITUDINAL=df_P_i['X_mm']*np.cos(alphaP[i-1])+df_P_i['Y_mm']*np.sin(alphaP[i-1]))
        df_P_i = df_P_i.assign(
            TRANVERSAL=-df_P_i['X_mm']*np.sin(alphaP[i-1])+df_P_i['Y_mm']*np.cos(alphaP[i-1]))
        f = pd.pivot_table(df_P_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)

        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        x1_cr, y1_cr = correction(eps_L[i-1], x, y1)
        x2_cr, y2_cr = correction(eps_T[i-1], x, y2)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x1_cr, y1_cr, color='r', label='Longitudinal', linestyle='-')
        ax.plot(x2_cr, y2_cr, color='g', label='Traversal', linestyle='-')
        ax.hlines(y=long_Pos[i-1], color='b', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Positive')
        ax.hlines(y=long_Neg[i-1], color='g', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Negative')
        ax.hlines(y=tran_Pos[i-1], color='r', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Positive')
        ax.hlines(y=tran_Neg[i-1], color='orange', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Negative')
        ax.set_title('P_XY_cor-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass
    return fig
# %%D_XYcor_Plot Funtion to plot after correction


def D_XY_cor_Plot(df, i):
    alphaD = np.deg2rad(df_design_limits['D']['ANGLE'])
    epsilon_dx = df_design_limits['D']['EPSILON X']
    epsilon_dy = df_design_limits['D']['EPSILON Y']

    # calculate epsilon for Longitudinal and Tranversal
    eps_L = epsilon_dx*np.cos(alphaD)+epsilon_dy*np.sin(alphaD)
    eps_T = -epsilon_dx*np.sin(alphaD)+epsilon_dy*np.cos(alphaD)

    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        df_D_i = df_D_i.assign(
            LONGITUDINAL=df_D_i['X_mm']*np.cos(alphaD[i-1])+df_D_i['Y_mm']*np.sin(alphaD[i-1]))
        df_D_i = df_D_i.assign(
            TRANVERSAL=-df_D_i['X_mm']*np.sin(alphaD[i-1])+df_D_i['Y_mm']*np.cos(alphaD[i-1]))
        f = pd.pivot_table(df_D_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)

        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        x1_cr, y1_cr = correction(eps_L[i-1], x, y1)
        x2_cr, y2_cr = correction(eps_T[i-1], x, y2)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        #ax.plot(x,y1, color='r', label='Longitudinal',linestyle='-.')
        #ax.plot(x,y2, color='g', label='Traversal',linestyle='-.')
        ax.plot(x1_cr, y1_cr, color='r',
                label='Longitudinal-cor', linestyle='-')
        ax.plot(x2_cr, y2_cr, color='g', label='Traversal-cor', linestyle='-')

        ax.set_title('D_XY_cor-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        fig = None
        pass

    return fig

 # %%


# %%Pzcor_Plot Funtion to compare

def Pzcpr_Plot(df, i):
    epsilon_pz = df_design_limits['P']['EPSILON Z']
    settlement_pz = df_design_limits['P']['SETTLEMENT']
    try:
        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        f = pd.pivot_table(
            df_P_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        x_cr, y_cr = correction(epsilon_pz[i-1], x, y)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x, y, color='r', label='Z', linestyle=':')
        ax.plot(x_cr, y_cr, color='g', label='Z-correction')
        ax.hlines(y=settlement_pz[i-1], color='b', xmin=x_cr[0], xmax=x_cr[len(
            x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('P_Z-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass

    return fig


# %%Dzcpr_Plot Funtion to compare
def Dzcpr_Plot(df, i):
    epsilon_dz = df_design_limits['D']['EPSILON Z']
    settlement_dz = df_design_limits['D']['SETTLEMENT']
    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        f = pd.pivot_table(
            df_D_i, values=['X_mm', 'Y_mm', 'Z_mm'], index='Date', aggfunc=np.mean)
        x = f.reset_index()["Date"]
        y = f.reset_index()["Z_mm"]
        x_cr, y_cr = correction(epsilon_dz[i-1], x, y)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x, y, color='r', label='Z', linestyle='-')

        ax.plot(x_cr, y_cr, color='g', label='Z-correction')

        ax.hlines(y=settlement_dz[i-1], color='b', xmin=x[0], xmax=x[len(
            x_cr)-1], linestyle='--', lw=2, label='Settlement Design Value')
        ax.set_title('D_Z-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        fig = None
        pass

    return fig

# %%P_XYcor_Plot Funtion to compare


def P_XYcpr_Plot(df, i):

    alphaP = np.deg2rad(df_design_limits['P']['ANGLE'])
    epsilon_px = df_design_limits['P']['EPSILON X']
    epsilon_py = df_design_limits['P']['EPSILON Y']

    # calculate epsilon for Longitudinal and Tranversal
    eps_L = epsilon_px*np.cos(alphaP)+epsilon_py*np.sin(alphaP)
    eps_T = -epsilon_px*np.sin(alphaP)+epsilon_py*np.cos(alphaP)

    long_Pos = df_design_limits['P']['TOP LONGITUDINAL POSITIVE']
    long_Neg = df_design_limits['P']['TOP LONGITUDINAL NEGATIVE']
    tran_Pos = df_design_limits['P']['TOP TRANSVERSAL POSITIVE']
    tran_Neg = df_design_limits['P']['TOP TRANSVERSAL NEGATIVE']

    try:

        df_P_i = df[(df['Pier'] == 'P') & (df['No'] == i)]
        df_P_i = df_P_i.assign(
            LONGITUDINAL=df_P_i['X_mm']*np.cos(alphaP[i-1])+df_P_i['Y_mm']*np.sin(alphaP[i-1]))
        df_P_i = df_P_i.assign(
            TRANVERSAL=-df_P_i['X_mm']*np.sin(alphaP[i-1])+df_P_i['Y_mm']*np.cos(alphaP[i-1]))
        f = pd.pivot_table(df_P_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)

        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        x1_cr, y1_cr = correction(eps_L[i-1], x, y1)
        x2_cr, y2_cr = correction(eps_T[i-1], x, y2)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.plot(x, y1, color='orange', linestyle='-', label='Longitudinal')
        ax.plot(x, y2, color='blue', linestyle='-', label='Traversal')
        ax.plot(x1_cr, y1_cr, color='r',
                label='Longitudinal-cor', linestyle='-')
        ax.plot(x2_cr, y2_cr, color='g', label='Traversal-cor', linestyle='-')
        ax.hlines(y=long_Pos[i-1], color='b', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Positive')
        ax.hlines(y=long_Neg[i-1], color='g', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Longitudinal Negative')
        ax.hlines(y=tran_Pos[i-1], color='r', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Positive')
        ax.hlines(y=tran_Neg[i-1], color='orange', xmin=x[0], xmax=x[len(x)-1],
                  linestyle='--', lw=2, label='Top Tranversal Negative')
        ax.set_title('P_XY-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        pass
    return fig
# %%D_XYcor_Plot Funtion to compare


def D_XYcpr_Plot(df, i):
    alphaD = np.deg2rad(df_design_limits['D']['ANGLE'])
    epsilon_dx = df_design_limits['D']['EPSILON X']
    epsilon_dy = df_design_limits['D']['EPSILON Y']

    # calculate epsilon for Longitudinal and Tranversal
    eps_L = epsilon_dx*np.cos(alphaD)+epsilon_dy*np.sin(alphaD)
    eps_T = -epsilon_dx*np.sin(alphaD)+epsilon_dy*np.cos(alphaD)

    try:
        df_D_i = df[(df['Pier'] == 'D') & (df['No'] == i)]
        df_D_i = df_D_i.assign(
            LONGITUDINAL=df_D_i['X_mm']*np.cos(alphaD[i-1])+df_D_i['Y_mm']*np.sin(alphaD[i-1]))
        df_D_i = df_D_i.assign(
            TRANVERSAL=-df_D_i['X_mm']*np.sin(alphaD[i-1])+df_D_i['Y_mm']*np.cos(alphaD[i-1]))
        f = pd.pivot_table(df_D_i, values=[
                           'X_mm', 'Y_mm', 'Z_mm', 'LONGITUDINAL', 'TRANVERSAL'], index='Date', aggfunc=np.mean)

        x = f.reset_index()["Date"]
        y1 = f.reset_index()["LONGITUDINAL"]
        y2 = f.reset_index()["TRANVERSAL"]

        x1_cr, y1_cr = correction(eps_L[i-1], x, y1)
        x2_cr, y2_cr = correction(eps_T[i-1], x, y2)

        #fig, ax = plt.subplots(ncols=1, nrows=1)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(x, y1, color='orange', label='Longitudinal', linestyle='-')
        ax.plot(x, y2, color='b', label='Traversal', linestyle='-')
        ax.plot(x1_cr, y1_cr, color='r',
                label='Longitudinal-cor', linestyle='-')
        ax.plot(x2_cr, y2_cr, color='g', label='Traversal-cor', linestyle='-')

        ax.set_title('D_XY-'+str(i))
        ax.set_xlabel('Date')
        ax.set_ylabel('mm')
        ax.legend(loc='best')
        fig.autofmt_xdate()
    except:
        fig = None
        pass

    return fig



    






