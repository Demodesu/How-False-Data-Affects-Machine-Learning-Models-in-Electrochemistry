import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os 
import sklearn
import shap 
import time
import math
import seaborn as sns
import scipy.stats as sps
import pathlib
import sys

font_size_plot = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

plt.gca().ticklabel_format(axis='both',style='plain',useOffset=False)

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

model_list = ['STACK','DEC','ELAS','XGB','LGBM','NN','LASS','RIDGE','RF','SVM','KNN','GB','ADA']
randomize_list = ['FEATURE','TARGET','FEATURETARGET']
random_value_list = [0,0.5,1]

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

max_row = 2
max_col = 4
row_num = 0
col_num = -1

fig = plt.figure(figsize=(20,11.25))

for model in model_list:

    for randomize in randomize_list:

        df_list = []

        for random_value in random_value_list:

            df = pd.read_csv(f'{randomize}\\PDPMODEL{model}RANDOM{randomize}VALUE{random_value}.csv')
            df_list.append(df)

        if model == 'DEC':

            fig.suptitle(f'DT PDP')

        else:

            fig.suptitle(f'{model} PDP')

        ax_label = 'abcdefghijklmnopqrstuvwxyz'

        for label_count,col in enumerate(df.columns):

            if col != 'FEATURE_VALUE':

                col_num += 1

                if col_num == max_col:

                    col_num = 0

                    row_num += 1

                ax = plt.subplot2grid((max_row,max_col),(row_num,col_num),fig=fig)

                ax.plot(df['FEATURE_VALUE'],df_list[0][col],c='red',linewidth=3,label='PDP 0% Noise')    
                ax.plot(df['FEATURE_VALUE'],df_list[1][col],c='blue',linewidth=3,label='PDP 50% Noise')
                ax.plot(df['FEATURE_VALUE'],df_list[2][col],c='green',linewidth=3,label='PDP 100% Noise')  

                ax.set_ylabel(f'Scaled CAP')
                ax.set_xlabel(f'Scaled {col}')

                ax.text(x=-0.15,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

        red_patch = matplotlib.patches.Patch(facecolor='red',label='PDP 0% Noise',linewidth=1.5,edgecolor='black')
        blue_patch = matplotlib.patches.Patch(facecolor='blue',label='PDP 50% Noise',linewidth=1.5,edgecolor='black')
        green_patch = matplotlib.patches.Patch(facecolor='green',label='PDP 100% Noise',linewidth=1.5,edgecolor='black')

        patch_list = [red_patch,blue_patch,green_patch]

        fig.tight_layout(rect=[0.05,0.05,0.95,0.95])
        fig.legend(handles=patch_list,loc='lower center',shadow=True,ncol=5,prop={'size':17.5},columnspacing=0.8)

        fig.savefig(f'Plot\\PDP{model}RANDOM{randomize}.jpg',dpi=fig.dpi)

        row_num = 0
        col_num = -1

        fig.clf()
