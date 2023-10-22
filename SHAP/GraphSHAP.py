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

font_size_plot = 22
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
noise_list = [0,0.5,1]
              
pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

original_df = pd.read_csv(f'CSV_SHAP\\Original.csv')

interest = '%N'

for model in model_list:

    plot_df_dict = {'Noise 0%':pd.DataFrame(),'Noise 50%':pd.DataFrame(),'Noise 100%':pd.DataFrame()}

    noise_name = -50

    # append df to dict

    for noise in noise_list:

        noise_name += 50
        key_name = f'Noise {noise_name}%'
        df = pd.read_csv(f'CSV_SHAP\\SHAP{model}{noise}.csv')
        plot_df_dict[key_name] = df

    # find max and min

    max_list = []
    min_list = []

    for key,value in plot_df_dict.items():

        max_value = value[interest].max()
        min_value = value[interest].min()
        max_list.append(max_value)
        min_list.append(min_value)

    offset = 0.05
    all_df_max = max(max_list)+offset
    all_df_min = min(min_list)-offset

    # plot

    fig = plt.figure(figsize=(20,11.25/2))

    if model == 'DEC':

        fig.suptitle(f'Noise Effect DT Model (Featuretarget) SHAP')

    else:

        fig.suptitle(f'Noise Effect {model} Model (Featuretarget) SHAP')

    ax_count = -1        
    label_count = -1
    title_name_noise = -50

    for key,value in plot_df_dict.items():

        title_name_noise += 50
        title_name = f'Noise {title_name_noise}%'

        ax_count += 1

        ax = plt.subplot2grid((1,3),(0,ax_count),fig=fig) # (rows,cols)

        sns.scatterplot(x=original_df[interest],y=value[interest],legend=False,c=value[interest],cmap='gist_rainbow',marker='o',s=100,norm=plt.Normalize(vmin=all_df_min,vmax=all_df_max))

        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.axhline(0,linestyle='--')
        ax.axvline(0,linestyle='--')
        
        ax.set_box_aspect(1)

        ax.set_ylim(all_df_min,all_df_max)

        label_count += 1
        
        ax_label = 'abcdefghijklmnopqrstuvwxyz'
        ax.text(x=-0.1,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

        ax.set_title(f'{title_name}')

    fig.tight_layout(rect=[0,0,0.82,1])

    sm = plt.cm.ScalarMappable(cmap='gist_rainbow',norm=plt.Normalize(vmin=all_df_min,vmax=all_df_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85,0.1,0.05,0.75]) # [left,bottom,right,top]
    fig.colorbar(sm,cax=cbar_ax)

    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    plt.xlabel(f'{interest}')
    plt.ylabel(f'SHAP {interest}',labelpad=10)

    fig.savefig(f'Plot\\{model}.jpg',dpi=fig.dpi)

    plt.close()