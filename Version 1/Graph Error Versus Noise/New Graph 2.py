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

font_size_plot = 19
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

model_list = ['LGBM','XGB','GB','ADA','RF','DEC','ELAS','LASS','RIDGE','NN','SVM','KNN','STACK']
randomize_list = ['FEATURE','TARGET','FEATURETARGET']

fig = plt.figure(figsize=(20,28))

row_index = -1
col_index = -1

for randomize_type in randomize_list:

    # fig.suptitle(f'Perturbed {randomize_type[0]+randomize_type[1:].lower()} Supercapacitor Dataset',fontsize=30)

    for ax_count,model in enumerate(model_list):

        # row_index += 1

        # if row_index % 4 == 0:

        #     row_index = 0

        #     col_index += 1

        col_index += 1

        if col_index % 3 == 0:

            col_index = 0

            row_index += 1

        ax = plt.subplot2grid((5,3),(row_index,col_index),fig=fig) # (rows,cols)

        df = pd.read_csv(f'CSV_models\\{randomize_type}{model}.csv')
        noise_df = pd.DataFrame()
        avg_df = pd.DataFrame()
        time_df = pd.DataFrame()
        trend_df = pd.DataFrame(columns=['NOISE','AVG'])

        for column in df:
            
            if 'NOISE' in column:

                noise_df = pd.concat([noise_df,df[column]],axis='columns')

            if 'AVG' in column:
            
                avg_df = pd.concat([avg_df,df[column]],axis='columns')   

            if 'TIME' in column:
                
                time_df = pd.concat([time_df,df[column]],axis='columns')  

        noise_series_list = []

        for column in noise_df:

            noise_series_list.append(noise_df[column])

        trend_df['NOISE'] = pd.concat(noise_series_list,axis='rows')

        avg_series_list = []

        for column in avg_df:
            
            avg_series_list.append(avg_df[column])

        trend_df['AVG'] = pd.concat(avg_series_list,axis='rows')

        trend_df = trend_df.reset_index(drop=True)

        noise_df = noise_df.transpose().reset_index(drop=True)
        avg_df = avg_df.transpose().reset_index(drop=True)

        column_str_list = [f'{x:.2f}' for x in np.arange(0,1.1,0.1)]
        column_float_list = [float(x) for x in column_str_list]
        avg_df.columns = column_str_list

        if model in ['LGBM','XGB','GB','ADA','RF','DEC']:
    
            if model == 'DEC':

                model = 'DT'

            model_group = 'Tree Based'
            
            ax.set_title(f'{model} ({model_group})',fontsize=22)

            sns.boxplot(data=avg_df,ax=ax,linewidth=2)
            sns.stripplot(data=avg_df,ax=ax,edgecolor='black',linewidth=2,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=5,ax=ax,color='red')

            ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=22)
            ax.set_xlabel('')
            ax.legend(loc='upper left',fontsize=18)

            ax.set_ylim(0,200)

            ax.set_box_aspect(1)

            # ax.set_xlabel('Noise (%)',fontsize=22,labelpad=20)
            # ax.set_ylabel('Error (F/g)',fontsize=22)

        elif model in ['ELAS','LASS','RIDGE']:

            model_group = 'Linear'

            ax.set_title(f'{model} ({model_group})',fontsize=22)

            sns.boxplot(data=avg_df,ax=ax,linewidth=2)
            sns.stripplot(data=avg_df,ax=ax,edgecolor='black',linewidth=2,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=5,ax=ax,color='red')

            ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=22)
            ax.set_xlabel('')
            ax.legend(loc='upper left',fontsize=18)

            ax.set_ylim(0,200)

            ax.set_box_aspect(1)

            # ax.set_xlabel('Noise (%)',fontsize=22,labelpad=20)
            # ax.set_ylabel('Error (F/g)',fontsize=22)

        elif model in ['KNN','SVM','NN']:
    
            model_group = 'Misc.'

            ax.set_title(f'{model} ({model_group})',fontsize=22)

            sns.boxplot(data=avg_df,ax=ax,linewidth=2)
            sns.stripplot(data=avg_df,ax=ax,edgecolor='black',linewidth=2,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=5,ax=ax,color='red')

            ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=22)
            ax.set_xlabel('')
            ax.legend(loc='upper left',fontsize=18)

            ax.set_ylim(0,200)

            ax.set_box_aspect(1)

            # ax.set_xlabel('Noise (%)',fontsize=22,labelpad=20)
            # ax.set_ylabel('Error (F/g)',fontsize=22)

        else:

            model_group = 'Stack'

            ax.set_title(f'{model} ({model_group})',fontsize=22)

            sns.boxplot(data=avg_df,ax=ax,linewidth=2)
            sns.stripplot(data=avg_df,ax=ax,edgecolor='black',linewidth=2,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=5,ax=ax,color='red')

            ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=22)
            ax.set_xlabel('')
            ax.legend(loc='upper left',fontsize=18)

            ax.set_ylim(0,200)

            ax.set_box_aspect(1)

            # ax.set_xlabel('Noise (%)',fontsize=22,labelpad=20)
            # ax.set_ylabel('Error (F/g)',fontsize=22)

        ax_label = 'abcdefghijklmnopqrstuvwxyz'
        ax.text(x=-0.15,y=1.1,s=f'{ax_label[ax_count]})',transform=ax.transAxes,size=font_size_plot+5)

    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    plt.xlabel('Noise (%)',fontsize=30,labelpad=30)
    plt.ylabel('Error (F/g)',fontsize=30,labelpad=30)

    # ax_tree.text(-0.07,1.07,'a)',transform=ax_tree.transAxes,size=font_size_plot+5)
    # ax_linear.text(-0.07,1.07,'b)',transform=ax_linear.transAxes,size=font_size_plot+5)
    # ax_misc.text(-0.07,1.07,'c)',transform=ax_misc.transAxes,size=font_size_plot+5)
    # ax_stack.text(-0.07,1.07,'d)',transform=ax_stack.transAxes,size=font_size_plot+5)

    # plt.subplots_adjust(hspace=0.7,wspace=0.3)

    # fig.add_subplot(111,frameon=False)
    # plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    # plt.xlabel('Noise (%)',labelpad=80,fontsize=22)
    # plt.ylabel('Error (F/g)',labelpad=10,fontsize=22)

    fig.tight_layout(rect=[0.05,0,0.95,1])

    fig.savefig(f'{randomize_type}2.jpg',dpi=fig.dpi)

    row_index = -1
    col_index = -1

    fig.clf()
