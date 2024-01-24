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

font_size_plot = 18
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
seed_list = [x for x in range(10)]

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

all_dataset_zmax = 0
all_dataset_zmin = 0

for model in model_list:

    for randomize in randomize_list:

        # append all noise and seed

        concat_df_dict = {'noise0':pd.DataFrame(),'noise50':pd.DataFrame(),'noise100':pd.DataFrame()}
        concat_count = 0

        for key,value in concat_df_dict.items():

            temp_df = pd.DataFrame()

            if concat_count == 1.0:

                concat_count = 1

            for seed in seed_list:

                df = pd.read_csv(f'{randomize}\\MODEL{model}RANDOMIZE{randomize}NOISE{concat_count}SEED{seed}.csv')

                temp_df = pd.concat([temp_df,df],axis='rows')

            temp_df = temp_df.reset_index(drop=True)
            concat_count += 0.5
            concat_df_dict[key] = temp_df

        # find max of all dataset for colorbar

        for key,value in concat_df_dict.items():

            if all_dataset_zmax < np.max(value['REAL CAP']):

                all_dataset_zmax = np.max(value['REAL CAP'])

            if all_dataset_zmax < np.max(value['PREDICTION CAP']):

                all_dataset_zmax = np.max(value['PREDICTION CAP'])     

for model in model_list:

    for randomize in randomize_list:

        print(model,randomize)

        # append all noise and seed

        concat_df_dict = {'noise0':pd.DataFrame(),'noise50':pd.DataFrame(),'noise100':pd.DataFrame()}
        concat_count = 0

        for key,value in concat_df_dict.items():

            temp_df = pd.DataFrame()

            if concat_count == 1.0:

                concat_count = 1

            for seed in seed_list:

                df = pd.read_csv(f'{randomize}\\MODEL{model}RANDOMIZE{randomize}NOISE{concat_count}SEED{seed}.csv')

                temp_df = pd.concat([temp_df,df],axis='rows')

            temp_df = temp_df.reset_index(drop=True)
            concat_count += 0.5
            concat_df_dict[key] = temp_df

        # plot first graph

        fig = plt.figure(figsize=(20,11.25))

        if model == 'DEC':

            fig.suptitle(f'Noise Effect DT Model ({randomize[0]+randomize[1:].lower()})')
        
        else:

            fig.suptitle(f'Noise Effect {model} Model ({randomize[0]+randomize[1:].lower()})')       

        ax_list = []

        ax = plt.subplot2grid((2,2),(0,0),fig=fig)
        ax.set_title(f'True Contour')

        ax.set_xlabel('%N')
        ax.set_ylabel('%O')

        ax_list.append(ax)

        # plot on axis

        x = concat_df_dict['noise0']['%N'].values
        y = concat_df_dict['noise0']['%O'].values

        z = concat_df_dict['noise0']['REAL CAP'].values
        z_scaled = (z - z.min()) / (z.max() - z.min())

        values = np.vstack([x,y]) # the values of x and y axis
        kernel = sps.gaussian_kde(values,weights=z_scaled) # estimate a gaussian kernel density estimation for each x and y value

        x_axis_range,y_axis_range = np.mgrid[x.min():x.max():200j,y.min():y.max():200j] # mesh grid of 100 points ranging from min to max
        x_and_y_positions = np.vstack([x_axis_range.ravel(),y_axis_range.ravel()]) # stack arrays vertically, position of x and y axis
        
        kde_space = np.reshape(kernel(x_and_y_positions).T,x_axis_range.shape) # reshap the KDE to x shape
        kde_space = np.rot90(kde_space)

        kde_space_df = pd.DataFrame(kde_space)
        kde_space_min = min(kde_space_df.min()) # find the true minimum of kde
        kde_space_max = max(kde_space_df.max())
        min_max_scale_kde_space_df = (kde_space_df.copy() - kde_space_min) / (kde_space_max - kde_space_min)
        capacitance_scale_z_df = min_max_scale_kde_space_df * z.max() # from (x * (xmax-xmin)) + xmin where xmin = 0

        # normalize over all the datasets to show difference in height of target variable
        ax.imshow(capacitance_scale_z_df,cmap='gist_rainbow',extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto',norm=plt.Normalize(vmin=all_dataset_zmin,vmax=all_dataset_zmax))

        # plot the rest

        max_row = 2
        max_col = 2
        row_count = 0
        col_count = 0
        noise_title_amount = -50

        for key,value in concat_df_dict.items():

            # decoration and axis

            noise_title_amount += 50
            noise_title = f'Noise {noise_title_amount}%'

            col_count += 1

            if col_count == max_col:

                col_count = 0
                row_count += 1

            ax = plt.subplot2grid((max_row,max_col),(row_count,col_count),fig=fig)
            ax.set_title(f'{noise_title}')

            ax.set_xlabel('%N')
            ax.set_ylabel('%O')

            ax_list.append(ax)

            # plot on axis

            x = value['%N'].values
            y = value['%O'].values

            z = value['PREDICTION CAP'].values
            z_scaled = (z - z.min()) / (z.max() - z.min())

            values = np.vstack([x,y]) # the values of x and y axis
            kernel = sps.gaussian_kde(values,weights=z_scaled) # estimate a gaussian kernel density estimation for each x and y value

            x_axis_range,y_axis_range = np.mgrid[x.min():x.max():200j,y.min():y.max():200j] # mesh grid of 100 points ranging from min to max
            x_and_y_positions = np.vstack([x_axis_range.ravel(),y_axis_range.ravel()]) # stack arrays vertically, position of x and y axis
            
            kde_space = np.reshape(kernel(x_and_y_positions).T,x_axis_range.shape) # reshap the KDE to x shape
            kde_space = np.rot90(kde_space)

            kde_space_df = pd.DataFrame(kde_space)
            kde_space_min = min(kde_space_df.min()) # find the true minimum of kde
            kde_space_max = max(kde_space_df.max())
            min_max_scale_kde_space_df = (kde_space_df.copy() - kde_space_min) / (kde_space_max - kde_space_min)
            capacitance_scale_z_df = min_max_scale_kde_space_df * z.max() # from (x * (xmax-xmin)) + xmin where xmin = 0

            # normalize over all the datasets to show difference in height of target variable
            ax.imshow(capacitance_scale_z_df,cmap='gist_rainbow',extent=[x.min(),x.max(),y.min(),y.max()],aspect='auto',norm=plt.Normalize(vmin=all_dataset_zmin,vmax=all_dataset_zmax))

        # label

        label_count = -1

        for ax in ax_list:

            label_count += 1
            
            ax_label = 'abcdefghijklmnopqrstuvwxyz'
            ax.text(x=-0.1,y=1.1,s=f'{ax_label[label_count]})',transform=ax.transAxes,size=font_size_plot+5)

        # decoration

        fig.tight_layout(rect=[0,0,0.8,1])

        sm = plt.cm.ScalarMappable(cmap='gist_rainbow',norm=plt.Normalize(vmin=all_dataset_zmin,vmax=all_dataset_zmax))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
        fig.colorbar(sm,cax=cbar_ax)

        # save

        fig.savefig(f'Plot\\{model}{randomize}.jpg',dpi=fig.dpi)

        plt.close()
