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

model_list = ['DEC','RIDGE','STACK']
noise_list = [0,0.5,1]
              
pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

original_df = pd.read_csv(f'CSV_SHAP\\Original.csv')

print(original_df.columns)

for interest in original_df.columns:

    print(interest)

    true_max = 0
    true_min = 0

    for model in model_list:

        plot_df_dict = {'Noise 0%':pd.DataFrame(),'Noise 50%':pd.DataFrame(),'Noise 100%':pd.DataFrame()}

        noise_name = -50

        # append df to dict

        for noise in noise_list:

            noise_name += 50
            key_name = f'Noise {noise_name}%'
            df = pd.read_csv(f'CSV_SHAP\\SHAP{model}{noise}.csv')
            plot_df_dict[key_name] = df

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

        if all_df_max > true_max:

            true_max = all_df_max
        
        if all_df_min < true_min:

            true_min = all_df_min

    fig = plt.figure(figsize=(20,20))
    fig.suptitle('Noise (Featuretarget) Effect Model SHAP')

    for plot_ax_count in range (3):

        model = model_list[plot_ax_count]

        plot_df_dict = {'Noise 0%':pd.DataFrame(),'Noise 50%':pd.DataFrame(),'Noise 100%':pd.DataFrame()}

        noise_name = -50

        # append df to dict

        for noise in noise_list:

            noise_name += 50
            key_name = f'Noise {noise_name}%'
            df = pd.read_csv(f'CSV_SHAP\\SHAP{model}{noise}.csv')
            plot_df_dict[key_name] = df

        # plot

        ax_count = -1        
        label_count = -1
        title_name_noise = -50

        for key,value in plot_df_dict.items():

            title_name_noise += 50
            title_name = f'Noise {title_name_noise}%'

            ax_count += 1

            ax = plt.subplot2grid((3,3),(plot_ax_count,ax_count),fig=fig) # (rows,cols)

            sns.scatterplot(x=original_df[interest],y=value[interest],legend=False,c=value[interest],cmap='gist_rainbow',marker='o',s=150,norm=plt.Normalize(vmin=true_min,vmax=true_max))

            ax.set_xlabel(f'{interest}')
            ax.set_ylabel(f'SHAP {interest}')

            ax.axhline(0,linestyle='--')
            ax.axvline(0,linestyle='--')
            
            ax.set_box_aspect(1)

            ax.set_ylim(true_min,true_max)

            label_count += 1
            
            if model == 'DEC':

                ax_label = 'abc'
                ax.text(x=-0.15,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

            if model == 'RIDGE':

                ax_label = 'def'
                ax.text(x=-0.15,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

            if model == 'STACK':

                ax_label = 'ghi'
                ax.text(x=-0.15,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

            if model == 'DEC':

                ax.set_title(f'DT {title_name}')

            else:

                ax.set_title(f'{model} {title_name}')

    fig.tight_layout(rect=[0,0,0.82,1])

    sm = plt.cm.ScalarMappable(cmap='gist_rainbow',norm=plt.Normalize(vmin=true_min,vmax=true_max))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85,0.1,0.05,0.75]) # [left,bottom,right,top]
    fig.colorbar(sm,cax=cbar_ax)

    fig.savefig(f'Plot\\ALLMODEL{interest}.jpg',dpi=fig.dpi)

    plt.close()