#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:32:39 2022

@author: namnguyen
"""

#%%Input libraries 
import pandas as pd
import numpy as np
import re
import os,pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean 
from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import Functions_Data_Analysis_220908 as FDA




#%% Get the current directory path

path = pathlib.Path(os.getcwd())
path_input=path/'Input Files'
#path_input=pathlib.Path(path/'/Input Files')
files = os.listdir(path_input) # list of input files


#%% loads excel files(X,Y,Z) from input folder

files_xlsx = sorted([f for f in files if '.xlsx' in f])


#%%Input the expected design limits/Angles
#cr_path=pathlib.Path(path+'/Input Files/Design/')
cr_path=path/'Input Files/Design/'
df_design_limits = pd.read_excel (cr_path/'Design_Limits &Angles.xlsx', sheet_name=None,header=0, index_col=None)

for k,v in df_design_limits.items():
    v.drop(v.columns[0],axis=1, inplace=True)
    v.set_index(v.columns[0], inplace=True)


#%% Set each input file as a dictionarynmbbbnmmmmmmmmmmm                
file_dir_X=path_input/files_xlsx[0]
file_dir_Y=path_input/files_xlsx[1]
file_dir_Z=path_input/files_xlsx[2]
X_dict=pd.read_excel(file_dir_X, sheet_name=None, header=1,index_col=None)
Y_dict=pd.read_excel(file_dir_Y, sheet_name=None, header=1,index_col=None)
Z_dict=pd.read_excel(file_dir_Z, sheet_name=None, header=1,index_col=None)
del Y_dict['Chart']# delete unexpected sheet_name: "Chart"
#%% create X_dict_new

X_dict_new={}
for kx,vx in X_dict.items():
    kx_new=re.findall('[A-Z]+\w*-\w+-*\w*|P10_D\d+',kx)# refining of the sheet_name/ tab_name
    #k_new=re.findall(r'\d+ *[A-Z]+\w*-\w+-*\w*|\d+ *P10_D\d+',k)
    #kx_new=re.sub("(.*)\(Position Ter.*", "\g<1>", kx)
    vx.drop(vx.columns[1],axis=1,inplace=True)  
    vx.columns=["Date","X_mm"]
    vx.set_index("Date",inplace=True)   
    X_dict_new[kx_new[0]]= X_dict[kx]


#%%create Y_dict_new

Y_dict_new={}

for ky,vy in Y_dict.items():
    ky_new=re.findall('[A-Z]+\w*-\w+-*\w*|P10_D\d+',ky)# refining of the sheet_name/ tab_name
    vy.drop(vy.columns[1],axis=1,inplace=True)  
    vy.columns=["Date","Y_mm"]
    vy.set_index("Date",inplace=True) 
    Y_dict_new[ky_new[0]]= Y_dict[ky]
     

#%%create Z_dict_new

Z_dict_new={}
for kz,vz in Z_dict.items():
    kz_new=re.findall('[A-Z]+\w*-\w+-*\w*|P10_D\d+',kz)# refining of the sheet_name/ tab_name
    vz.drop(vz.columns[1],axis=1,inplace=True)  
    vz.columns=["Date","Z_mm"]
    vz.set_index(["Date"],inplace=True) 
    Z_dict_new[kz_new[0]]= Z_dict[kz]
     

#%% concatenate three dictionary to one and clean the abundant dataframes

xd=X_dict_new
yd=Y_dict_new
zd=Z_dict_new

ds=[xd,yd,zd]
d={}
for k in xd.keys():
   d[k]=pd.concat(list(d[k]for d in ds),axis=1)
del d["P-P12-1"]# delete the duplicated tab 
del d['P7-04-P001']# delete the redundant tab



#%% Reshape the keys in dictionary and 

pt='([A-Z]+)-?(\d+)-?(\d*)'

main_dict=dict()# initial an empty dictionary
P10_dict={}#  dictionary which will contain the new data for some points
for k in d.keys():
	k1=re.sub('P10_','',k)
	B=re.match(pt,k1)
	## let's use list comprehension to replace all the missing numbers with 0.
	B_digits=['0' if j=='' else str(int(j)) for j in B.groups()[1:]]
	g_type=B[1]
	g_No1=B_digits[0]
	g_No2=B_digits[1]
	k_new=g_type+"-"+g_No1+"-"+g_No2
	#print(B[0]+" --> "+k_new)
	if "P10_" in k: # to identify the new datas of points
		P10_dict[k_new]=d[k]# new datas save into P10_dict
	else:
		main_dict[k_new]=d[k] #Old datas are saved in a main_dict
       

#%% Joint and update data in main dictionary
# New data from P10_dict is joint in main_dict when point name is the same
for k in P10_dict.keys():
    if k in main_dict.keys():
        main_dict[k]=pd.concat([main_dict[k],P10_dict[k]]).sort_index()
    else:
        main_dict[k]=P10_dict[k].sort_index()
#%% reset the data of the first line equal to 0 at the begining

for k in main_dict.keys(): 
    main_dict[k].iloc[:]=main_dict[k].iloc[:]-main_dict[k].iloc[0]
    

#%%In Pier P-17-2 and P-17-4 there are some strange dispersions. We elimilate them 


epsilon_px = df_design_limits['P']['EPSILON X']
epsilon_py = df_design_limits['P']['EPSILON Y']
epsilon_pz = df_design_limits['P']['EPSILON Z']

K=["P-17-2","P-17-4"]
for k in K:
    main_dict[k]["X_mm"]=FDA.dispersion(main_dict[k]["X_mm"],epsilon_px[16])
    main_dict[k]["Y_mm"]=FDA.dispersion(main_dict[k]["Y_mm"],epsilon_py[16])
    main_dict[k]["Z_mm"]=FDA.dispersion(main_dict[k]["Z_mm"],epsilon_py[16])
    



#%%
import copy
D_ex=copy.deepcopy(main_dict)
# =============================================================================
# #change to format date// I do not recommend to change formatdate
# for k in main_dict.keys(): 
#     D_ex[k].index=pd.to_datetime(D_ex[k].index).strftime('%Y-%m-%d')
#     
# =============================================================================
os.makedirs(path/'Output Files',exist_ok=True)
path_output=path/'Output Files'

with pd.ExcelWriter(path_output/'XYZ_joined.xlsx') as writer:
    for k, v in D_ex.items():
        v.to_excel(writer, sheet_name=k)
       


#%% Reset index in main dictionary
#For each point the dictionary key is divided into three catergories: 
    #"Pier": P or D
    #"No": Pier/Mid span nnumber
    #"Point": points within each pier
    
    
a=copy.deepcopy(main_dict)
pt='([A-Z]+)-?(\d+)-?(\d*)'

for k in a.keys():
    B=re.match(pt,k)
    a[k]['Pier']=B[1]
    a[k]['Point']=int(B[3])
    a[k]['No']=int(B[2])
    a[k]=a[k].reset_index(drop=False)
    a[k]=a[k]
    a[k].set_index(["Pier","No", "Point", "Date"], inplace=True)
       

#%% convert dictionary to one dataframe

df=pd.concat(a.values())
df=df.reset_index(drop=False)  
df=df.convert_dtypes()
    
        
#%% Seaborn to get multiplot 
# # all graphics in one figure:
#fig=sns.relplot(data=df ,x='Date',y="Z_mm",estimator=np.mean,col="Pier", row="No",ci=None,facet_kws={'sharey': False, 'sharex': True},kind='line')



              
 

#%% If you want to plot each single point in each pier, just access to the data frame:
    ## Single point Pz-i-j
for i in range(1,19):
    for j in range(1,5):
        try:

            fig=FDA.Pz_i_j(df, i, j)
            
            file_name="Pz-"+str(i)+"-"+str(j)+".png"
            folder_name='Graphics_without_correction/P_single_point/Pz'
            os.makedirs(path_output/folder_name ,exist_ok=True)
            plt.savefig(path_output/folder_name/file_name)
            plt.close(fig)
            
        except:
            pass

 

#%% If you want to plot each single point in each pier, just access to the data frame:
    ##single point Pxy-i-j
folder_name='Graphics_without_correction/P_single_point/Pxy'
os.makedirs(path_output/folder_name ,exist_ok=True)    
for i in range(1,19):
    for j in range(1,5):
        try:
        
            fig=FDA.Pxy_i_j(df, i, j)
            
            file_name="Pxy-"+str(i)+"-"+str(j)+".png"       
           
            fig.savefig(path_output/folder_name/file_name) 
            fig.clf()
            plt.close(fig)
        except:
             pass
        

#%%#######**************************
#%%Plot Dz in process

for i in range(1,18):
    fig=FDA.Dz_Plot(df, i)
    if fig!=None:
        file_name='D-'+str(i)+'.png'
        folder_name='Graphics_without_correction/Dz'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        fig.savefig(path_output/folder_name/file_name) 
        fig.clf()
        plt.close(fig)
    
#%%Plot Pz in process
for i in range(1,19):
    fig=FDA.Pz_Plot(df, i)
   
        
    file_name='Pz-'+str(i)+'.png'
    
    folder_name='Graphics_without_correction/Pz'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name) 
    
#%%Plot P_XY in process
for i in range(1,19):
    fig=FDA.P_XY_Plot(df, i)
    file_name='P_XY-'+str(i)+'.png'
    folder_name='Graphics_without_correction/Pxy'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name)    
    
#%%Plot D_XY in process
for i in range(1,18):
    fig=FDA.D_XY_Plot(df, i)
    if fig!=None:
        
        file_name='D_XY-'+str(i)+'.png'
        folder_name='Graphics_without_correction/Dxy'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        fig.savefig(path_output/folder_name/file_name)    
#%%





    
#%%Plopt Dz_cor after correction
for i in range(1,18):
    g=FDA.Dzcor_Plot(df, i)
    if g!=None:
        file_name='D_cor-'+str(i)+'.png'
        
        folder_name='Graphics_correction/Correction/Dz'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        g.savefig(path_output/folder_name/file_name) 
    
#%%Plot Pz_cor after correction
for i in range(1,19):
    fig=FDA.PzCor_Plot(df, i)
    
    file_name='P_cor-'+str(i)+'.png'
    
    folder_name='Graphics_correction/Correction/Pz'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name) 
    
#%%Plot P_XY_cor after correction
for i in range(1,19):
    fig=FDA.P_XYcor_Plot(df, i)
    file_name='P_XY_cor-'+str(i)+'.png'
    folder_name='Graphics_correction/Correction/Pxy'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name)    
    
#%%Plot D_XY_cor after correction
for i in range(1,18):
    fig=FDA.D_XY_cor_Plot(df, i)
    if fig!=None:
        
        file_name='D_XY_cor-'+str(i)+'.png'
        folder_name='Graphics_correction/Correction/Dxy'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        fig.savefig(path_output/folder_name/file_name)    



#%%





    
#%%Plopt Dz_cpr to compare
for i in range(1,18):
    g=FDA.Dzcpr_Plot(df, i)
    if g!=None:
        file_name='D_cpr-'+str(i)+'.png'
        
        folder_name='Graphics_correction/To Compare/Dz'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        g.savefig(path_output/folder_name/file_name) 
    
#%%Plot Pz_cpr to compare
for i in range(1,19):
    fig=FDA.Pzcpr_Plot(df, i)
    
    file_name='P_cpr-'+str(i)+'.png'
    
    folder_name='Graphics_correction/To Compare/Pz'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name) 
    
#%%Plot P_XY_cpr to compare
for i in range(1,19):
    fig=FDA.P_XYcpr_Plot(df, i)
    file_name='P_XY_cpr-'+str(i)+'.png'
    folder_name='Graphics_correction/To Compare/Pxy'
    os.makedirs(path_output/folder_name ,exist_ok=True)
    fig.savefig(path_output/folder_name/file_name)    
    
#%%Plot D_XY_cpr to compare
for i in range(1,18):
    fig=FDA.D_XYcpr_Plot(df, i)
    if fig!=None:
        
        file_name='D_XY_cpr-'+str(i)+'.png'
        folder_name='Graphics_correction/Correction/Dxy'
        os.makedirs(path_output/folder_name ,exist_ok=True)
        fig.savefig(path_output/folder_name/file_name)    


