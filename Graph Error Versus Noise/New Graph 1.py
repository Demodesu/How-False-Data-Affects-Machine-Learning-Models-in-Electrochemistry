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

font_size_plot = 25
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

prop_cycle = plt.rcParams['axes.prop_cycle']

for randomize_type in randomize_list:

    fig = plt.figure(figsize=(20,20))

    # fig.suptitle(f'Perturbed {randomize_type[0]+randomize_type[1:].lower()} Supercapacitor Dataset',fontsize=30)

    ax_tree = plt.subplot2grid((2,2),(0,0),fig=fig) # (rows,cols)
    ax_linear = plt.subplot2grid((2,2),(0,1),fig=fig) # (rows,cols)
    ax_misc = plt.subplot2grid((2,2),(1,0),fig=fig) # (rows,cols)
    ax_stack = plt.subplot2grid((2,2),(1,1),fig=fig) # (rows,cols)

    ax_list = [ax_tree,ax_linear,ax_misc,ax_stack]

    colors_list = prop_cycle.by_key()['color']
    colors_list.append('#0f2734')
    colors_list.append('#ccff00')
    colors_list.append('#00ff00')
    patch_list = []

    for model_count,model_type in enumerate(model_list):

        df = pd.read_csv(f'CSV_models\\{randomize_type}{model_type}.csv')
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

        # if model_type in ['LGBM','XGB','GB','ADA','RF','DEC']:
    
        #     if model_type == 'DEC':

        #         model_type = 'DT'

        #     model_group = 'Tree Based'
            
        #     ax_tree.set_title(f'{model_group}',fontsize=22)

        #     # sns.boxplot(data=avg_df,ax=ax_tree)
        #     # sns.stripplot(data=avg_df,ax=ax_tree,edgecolor='black',linewidth=1,jitter=False)  

        #     z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
        #     p = np.poly1d(z)
        #     sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'{model_type} {p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=4,ax=ax_tree)

        #     ax_tree.set_xticklabels(ax_tree.get_xticklabels(),rotation=90)
        #     ax_tree.set_xlabel('')
        #     ax_tree.legend(loc='upper left')

        #     ax_tree.set_ylim(0,100)

        #     ax_tree.set_box_aspect(1)

        #     ax_tree.set_xlabel('Noise',fontsize=22,labelpad=20)
        #     ax_tree.set_ylabel('Error (F/g)',fontsize=22)

        # elif model_type in ['ELAS','LASS','RIDGE']:

        #     model_group = 'Linear'

        #     ax_linear.set_title(f'{model_group}',fontsize=22)

        #     # sns.boxplot(data=avg_df,ax=ax_linear)
        #     # sns.stripplot(data=avg_df,ax=ax_linear,edgecolor='black',linewidth=1,jitter=False)  

        #     z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
        #     p = np.poly1d(z)
        #     sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'{model_type} {p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=4,ax=ax_linear)

        #     ax_linear.set_xticklabels(ax_linear.get_xticklabels(),rotation=90)
        #     ax_linear.set_xlabel('')
        #     ax_linear.legend(loc='upper left')

        #     ax_linear.set_ylim(0,100)

        #     ax_linear.set_box_aspect(1)

        #     ax_linear.set_xlabel('Noise',fontsize=22,labelpad=20)
        #     ax_linear.set_ylabel('Error (F/g)',fontsize=22)

        # elif model_type in ['KNN','SVM','NN']:
    
        #     model_group = 'Miscellaneous'

        #     ax_misc.set_title(f'{model_group}',fontsize=22)

        #     # sns.boxplot(data=avg_df,ax=ax_misc)
        #     # sns.stripplot(data=avg_df,ax=ax_misc,edgecolor='black',linewidth=1,jitter=False)  

        #     z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
        #     p = np.poly1d(z)
        #     sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'{model_type} {p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=4,ax=ax_misc)

        #     ax_misc.set_xticklabels(ax_misc.get_xticklabels(),rotation=90)
        #     ax_misc.set_xlabel('')
        #     ax_misc.legend(loc='upper left')

        #     ax_misc.set_ylim(0,100)

        #     ax_misc.set_box_aspect(1)

        #     ax_misc.set_xlabel('Noise',fontsize=22,labelpad=20)
        #     ax_misc.set_ylabel('Error (F/g)',fontsize=22)

        # else:

        #     model_group = 'Stack'

        #     ax_stack.set_title(f'{model_group}',fontsize=22)

        #     # sns.boxplot(data=avg_df,ax=ax_stack)
        #     # sns.stripplot(data=avg_df,ax=ax_stack,edgecolor='black',linewidth=1,jitter=False)  

        #     z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
        #     p = np.poly1d(z)
        #     sns.lineplot(x=column_str_list,y=p(column_float_list),label=f'{model_type} {p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=4,ax=ax_stack)

        #     ax_stack.set_xticklabels(ax_stack.get_xticklabels(),rotation=90)
        #     ax_stack.set_xlabel('')
        #     ax_stack.legend(loc='upper left')

        #     ax_stack.set_ylim(0,100)

        #     ax_stack.set_box_aspect(1)

        #     ax_stack.set_xlabel('Noise',fontsize=22,labelpad=20)
        #     ax_stack.set_ylabel('Error (F/g)',fontsize=22)

        if model_type in ['LGBM','XGB','GB','ADA','RF','DEC']:
    
            if model_type == 'DEC':

                model_type = 'DT'

            model_group = 'Tree Based'
            
            ax_tree.set_title(f'{model_group}')

            # sns.boxplot(data=avg_df,ax=ax_tree)
            # sns.stripplot(data=avg_df,ax=ax_tree,edgecolor='black',linewidth=1,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),linewidth=4,ax=ax_tree,color=colors_list[model_count])

            ax_tree.set_xticklabels(ax_tree.get_xticklabels(),rotation=90)
            ax_tree.set_xlabel('')

            ax_tree.set_ylim(0,100)

            # ax_tree.set_box_aspect(1)

            ax_tree.set_xlabel('Noise',labelpad=20)
            ax_tree.set_ylabel('Error (F/g)')

        elif model_type in ['ELAS','LASS','RIDGE']:

            model_group = 'Linear'

            ax_linear.set_title(f'{model_group}')

            # sns.boxplot(data=avg_df,ax=ax_linear)
            # sns.stripplot(data=avg_df,ax=ax_linear,edgecolor='black',linewidth=1,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),linewidth=4,ax=ax_linear,color=colors_list[model_count])

            ax_linear.set_xticklabels(ax_linear.get_xticklabels(),rotation=90)
            ax_linear.set_xlabel('')

            ax_linear.set_ylim(0,100)

            # ax_linear.set_box_aspect(1)

            ax_linear.set_xlabel('Noise',labelpad=20)
            ax_linear.set_ylabel('Error (F/g)')

        elif model_type in ['KNN','SVM','NN']:
    
            model_group = 'Miscellaneous'

            ax_misc.set_title(f'{model_group}')

            # sns.boxplot(data=avg_df,ax=ax_misc)
            # sns.stripplot(data=avg_df,ax=ax_misc,edgecolor='black',linewidth=1,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),linewidth=4,ax=ax_misc,color=colors_list[model_count])

            ax_misc.set_xticklabels(ax_misc.get_xticklabels(),rotation=90)
            ax_misc.set_xlabel('')

            ax_misc.set_ylim(0,100)

            # ax_misc.set_box_aspect(1)

            ax_misc.set_xlabel('Noise',labelpad=20)
            ax_misc.set_ylabel('Error (F/g)')

        else:

            model_group = 'Stack'

            ax_stack.set_title(f'{model_group}')

            # sns.boxplot(data=avg_df,ax=ax_stack)
            # sns.stripplot(data=avg_df,ax=ax_stack,edgecolor='black',linewidth=1,jitter=False)  

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)
            sns.lineplot(x=column_str_list,y=p(column_float_list),linewidth=4,ax=ax_stack,color=colors_list[model_count])

            ax_stack.set_xticklabels(ax_stack.get_xticklabels(),rotation=90)
            ax_stack.set_xlabel('')

            ax_stack.set_ylim(0,100)

            # ax_stack.set_box_aspect(1)

            ax_stack.set_xlabel('Noise',labelpad=20)
            ax_stack.set_ylabel('Error (F/g)')

        legend_color = colors_list[model_count]

        patch = matplotlib.patches.Patch(facecolor=legend_color,label=f'{model_type} y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}',linewidth=1.5,edgecolor='black')
        patch_list.append(patch)

    ax_label = 'abcdefghijklmnopqrstuvwxyz'

    for ax_count,ax in enumerate(ax_list):

        ax.text(x=-0.05,y=1.1,s=f'{ax_label[ax_count]})',transform=ax.transAxes,size=font_size_plot+10)

    fig.legend(handles=patch_list,loc='upper center',shadow=True,ncol=5,prop={'size':17.5},columnspacing=0.8)

    # ax_tree.text(-0.07,1.07,'a)',transform=ax_tree.transAxes,size=font_size_plot+5)
    # ax_linear.text(-0.07,1.07,'b)',transform=ax_linear.transAxes,size=font_size_plot+5)
    # ax_misc.text(-0.07,1.07,'c)',transform=ax_misc.transAxes,size=font_size_plot+5)
    # ax_stack.text(-0.07,1.07,'d)',transform=ax_stack.transAxes,size=font_size_plot+5)

    # plt.subplots_adjust(hspace=0.7,wspace=0.3)

    # fig.add_subplot(111,frameon=False)
    # plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
    # plt.xlabel('Noise (%)',labelpad=80,fontsize=22)
    # plt.ylabel('Error (F/g)',labelpad=10,fontsize=22)

    fig.tight_layout(rect=(0,0,1,0.93))

    fig.savefig(f'{randomize_type}.jpg',dpi=fig.dpi)

    plot_row = -1
    plot_col = 0

    fig.clf()
