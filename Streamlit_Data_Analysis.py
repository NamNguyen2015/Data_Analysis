#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:37:10 2022

@author: namnguyen
"""

#import numpy as np
#import math
import streamlit as st
#from math import pi, cos, sin
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from matplotlib.animation import PillowWriter
import pandas as pd
import os
#import base64
#from PIL import Image
import io
path= os.getcwd()

import final_Data_Analysis as DA
#%%
st.title('📊 Apply Data Science in Civil Engineering')
st.image("https://media-exp1.licdn.com/dms/image/C4E0BAQF-5O5stYOVnA/company-logo_200_200/0/1519880154681?e=2147483647&v=beta&t=JfMPNm2p8aQC7iHLqp8S4096lFDmShsodp8A73sRnWQ",width=100)
st.header('Company: CFC.SL' )
st.markdown("**Pedram Manouchehri** ")
st.markdown("**Nam Nguyen** ")
  
st.write('**************************************')

st.subheader("Introduction:")
#st.markdown('Streamlit is **_really_ cool**.')
st.markdown("""
*  **Python libraries:** pandas, streamlit, matplotlib
*  **Linked:** [Data_Analysis](https://github.com/NamNguyen2015/Data_Analysis)
             
""")
st.markdown("**Plotly Dash vs Streamlit — Which is the best?**")
st.markdown("[History of GitHub stars for both Plotly Dash and Streamlit](https://towardsdatascience.com/plotly-dash-vs-streamlit-which-is-the-best-library-for-building-data-dashboard-web-apps-97d7c98b938c)")
image = "https://upload.wikimedia.org/wikipedia/commons/7/77/Streamlit-logo-primary-colormark-darktext.png"

st.image(image, caption='The fastest way to build and share data apps')
st.write('Streamlit is an open source app framework in Python. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy, NumPy, pandas, Matplotlib etc.')

st.markdown("**About project**")
st.image(path+r'/High_Bridge.png', width=600)# Manually Adjust the width of the image as per requirement
st.markdown("**See image:** [Design image](https://github.com/NamNguyen2015/Data_Analysis/blob/main/monitoring/FFL-DE-DD-ST09-SUP1-SHP-DRA-0311%5B5.0%5D.pdf)")
st.markdown("**Previous public:** [High Bridge Monitoring](https://github.com/NamNguyen2015/Data_Analysis/blob/main/monitoring/High%20Bridge%20Monitoring.pdf)")
#st.image(path+r'/monitoring/FFL-DE-DD-ST09-SUP1-SHP-DRA-0311[5.0].jpeg', width=500)

st.write("In this project, we analize an excel dataset which collects the  movements of a  bridge from 2018 to July 2022.")
st.write("Our goal is to analyze the quality of the data, fix unacceptable dispersions and develop practical and sound plots.")
st.image(path+r'/DA_flow.jpg', width=600)

st.write('**************************************')

st.subheader("[Optional] Show the drop/drag file:")

uploaded_file = st.file_uploader("Upload EXCEL", type=".xlsx")

if uploaded_file is not None:

    dfm = pd.read_excel(uploaded_file,sheet_name=None)
  
    k_list=list(dfm.keys())
    N=len(k_list)
   
    st.write("This excel file contains " +str(N) + " sheet_names: ")
    
    for k in k_list[:3]: 
             
        st.write(k)
        
        st.write(dfm[k].head(10))
        

st.write('**************************************')
st.subheader("Show the final excel file")
D_ex=DA.D_ex
st.write("The file has "+str(len(D_ex.keys()))+" sheet_names")

option=list(D_ex.keys())

data=st.selectbox("which tab do you want to execute?", option,0)
   
st.write(D_ex[data].head(100))



st.line_chart(data=D_ex[data], x=None, y=None, width=0, height=0, use_container_width=True)


st.markdown("### Download Excel:")
   
buffer = io.BytesIO()    
   # Create some Pandas dataframes from data.   
#df1 = data
# Create a Pandas Excel writer using XlsxWriter as the engine.
   
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    # Write each dataframe to a different worksheet.
    for k, v in D_ex.items():
        v.to_excel(writer, sheet_name=k)
    #df1.to_excel(writer, sheet_name=new_data['Pier']+'-'+str(new_data['Numero']))
   
#writer.save()
Download_btn=st.download_button(
    label="📥  DOWNLOAD WHOLE FILE",
    data=buffer,
    file_name='D_ex.xlsx',
    mime='text/xlsx',
)

if Download_btn:
    st.success("Excel file is saved")



    

st.write('**************************************')  

st.subheader("Apply the  designed limits ") 

st.markdown("**Design limits and rotation angle**")   
dsa=DA.df_design_limits
for k, v in dsa.items():
    st.write(k)
    st.write(v)

  
        
st.markdown("**Rotation matrix**")

st.image("https://www.researchgate.net/profile/Adrian-Martinez-Vargas/publication/237030100/figure/fig3/AS:667659175739402@1536193760987/Definition-of-the-rotation-matrices-trough-the-axis-x-y-and-z-taken-from-Meyer-2000.ppm", width=300)


st.markdown("**Fixing dispersion signal-fundamental problem**")

st.text("")

st.markdown("""
            * [See published paper about thresholds](https://opg.optica.org/oe/fulltext.cfm?uri=oe-20-27-28319&id=246832)
            * [See document about step detection algorithm](https://dsp.stackexchange.com/questions/47227/need-a-better-step-detection-algorithm)
            
            """)

df=DA.df    

   
#st.sidebar.header("Options")
#st.header("Options")
#create selecboxes
#option_form=st.form("option_form")
PD=st.selectbox('Chose an option:', options=['D','P'])
Num=st.selectbox('Chose a number:', options=range(1,19))
#Num=option_form.selectbox('Chose a number:', options=range(1,19))

#add_data=option_form.form_submit_button("Generate")#the button to execute

st.markdown("**Data preview:**")
#if add_data:
st.write("Your data is: " +PD+str('-')+str(Num))  
new_data={'Pier': PD, 'Numero': Num}
    #st. write(new_data)  
    

data=df[(df["Pier"]==new_data['Pier'])&(df['No']==new_data['Numero'])]

current_data=st.write(data)

import Functions_Data_Analysis as fda


if new_data['Pier']=='D':
    fig1= fda.Dzcpr_Plot(df,new_data['Numero'] )
    fig2=fda.D_XYcpr_Plot(df, new_data['Numero'])
elif new_data['Pier']=='P':
    fig1= fda.Pzcpr_Plot(df,new_data['Numero'] )
    fig2=fda.P_XYcpr_Plot(df, new_data['Numero'])
st.write(fig1)
st.write(fig2)
        
st.markdown("### Download Current Data:")
   
buffer = io.BytesIO()    
   # Create some Pandas dataframes from data.   
df1 = data
# Create a Pandas Excel writer using XlsxWriter as the engine.
   
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    # Write each dataframe to a different worksheet.
    df1.to_excel(writer, sheet_name=new_data['Pier']+'-'+str(new_data['Numero']))
   
#writer.save()
Download_btn=st.download_button(
    label="📥  DOWNLOAD CURRENT DATA",
    data=buffer,
    file_name='data_analysis.xlsx',
    mime='text/xlsx',
)

if Download_btn:
    st.success("Excel file is saved")

###Save and download figures  
# =============================================================================
# 
# from zipfile import ZipFile
# from io import BytesIO
# 
# 
# zipObj = ZipFile("analysis_figures.zip", "w")
# # Add multiple files to the zip
# zipObj.write("hellostreamlit.txt")
# zipObj.write("hellostreamlit2.txt")
# #zipObj.write(fig1.savefig("figure1.png"))
# #zipObj.write(fig2.savefig("figure2.png"))
# # close the Zip File
# zipObj.close()
# 
# ZipfileDotZip = "analysis_figures.zip"
# 
# with open(ZipfileDotZip, "rb") as fp:
#     btn = st.download_button(
#         label="Download ZIP images",
#         data=fp,
#         file_name="analysis_figures.zip",
#         mime="application/zip"
#     )
# 
# 
# 
# =============================================================================


# ## To download image we can have two approaches: first, save to file and then download; second, save to memory and then download
# ##the second:

g=[fig1, fig2]

for i in range(1,len(g)+1):
    
    fn='fig'+str(i)            
    j=g[i-1].savefig(fn)    
    img = io.BytesIO()
    j.savefig(img)    
btn = st.download_button(
    label="DOWNLOAD IMAGE" +str(i),
     data=img,
    file_name=fn, )




