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
import pathlib

font_size_plot = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

fig = plt.figure(figsize=(20,12))

model_list = ['LGBM','XGB','NN','ELAS','LASS','RIDGE','RF','SVM','KNN','GB','ADA','DEC','STACK']

original_df = pd.read_csv('CSV\\original_df_validation_feature_seed.csv')
scale_df = pd.read_csv('CSV_scale_target_original\\scale_original_df_validation_feature_seed.csv')

for column in original_df:

    print(column)

    max_scale = scale_df[column].max()
    min_scale = scale_df[column].min()

    original_df[column] = original_df[column].apply(lambda x: (x * (max_scale - min_scale)) + min_scale)

row_index = -1
col_index = -1

pathlib.Path(f'Plot').mkdir(parents=True,exist_ok=True)

for model in model_list:

    fig.suptitle(f'{model} Model SHAP 0% Noise')

    SHAP_df = pd.read_csv(f'CSV_SHAP\\SHAP{model}.csv')

    for SHAP_count,column in enumerate(SHAP_df):

        col_index += 1

        if col_index % 3 == 0:

            col_index = 0

            row_index += 1

        print(row_index,col_index)

        interested_plot_original = original_df[column]
        interested_plot_SHAP = SHAP_df[column]

        ax = plt.subplot2grid((3,3),(row_index,col_index),fig=fig)

        sns.scatterplot(x=interested_plot_original,y=interested_plot_SHAP,legend=False,c=interested_plot_SHAP,cmap='icefire',marker='^',s=100)

        ax.set_ylabel(f'SHAP {column}')

        ax.axhline(0,linestyle='--')
        ax.axvline(0,linestyle='--')

        ax_label = 'hijklmnopqrstuvwxyz'
        ax.text(x=-0.1,y=1.1,s=f'{ax_label[SHAP_count]})',transform=ax.transAxes,size=font_size_plot+5)

        norm = plt.Normalize(interested_plot_SHAP.min(),interested_plot_SHAP.max())
        sm = plt.cm.ScalarMappable(cmap='icefire',norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm,ax=ax)

    row_index = -1
    col_index = -1

    fig.tight_layout()

    fig.savefig(f'Plot\\SHAP{model}.jpg',dpi=fig.dpi+100)

    fig.clf()
