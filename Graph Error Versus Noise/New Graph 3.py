# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import os 
# import sklearn
# import shap 
# import time
# import math
# import seaborn as sns
# import pathlib

# font_size_plot = 19
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['figure.labelweight'] = 'bold'
# plt.rcParams['figure.titleweight'] = 'bold'
# plt.rcParams['font.size'] = font_size_plot

# path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(f'{path}')

# model_list = ['LGBM','XGB','GB','ADA','RF','DEC','ELAS','LASS','RIDGE','NN','SVM','KNN','STACK']
# randomize_list = ['FEATURE','TARGET','FEATURETARGET']

# fig = plt.figure(figsize=(20,28))

# row_index = -1
# col_index = -1

# for randomize_type in randomize_list:

#     # fig.suptitle(f'Perturbed {randomize_type[0]+randomize_type[1:].lower()} Supercapacitor Dataset',fontsize=30)

#     for ax_count,model in enumerate(model_list):

#         # row_index += 1

#         # if row_index % 4 == 0:

#         #     row_index = 0

#         #     col_index += 1

#         col_index += 1

#         if col_index % 3 == 0:

#             col_index = 0

#             row_index += 1

#         ax = plt.subplot2grid((5,3),(row_index,col_index),fig=fig,projection='polar') # (rows,cols)

#         df = pd.read_csv(f'{randomize_type}{model}.csv')
#         noise_df = pd.DataFrame()
#         avg_df = pd.DataFrame()
#         time_df = pd.DataFrame()
#         trend_df = pd.DataFrame(columns=['NOISE','AVG'])

#         for column in df:
            
#             if 'NOISE' in column:

#                 noise_df = pd.concat([noise_df,df[column]],axis='columns')

#             if 'AVG' in column:
            
#                 avg_df = pd.concat([avg_df,df[column]],axis='columns')   

#             if 'TIME' in column:
                
#                 time_df = pd.concat([time_df,df[column]],axis='columns')  

#         noise_series_list = []

#         for column in noise_df:

#             noise_series_list.append(noise_df[column])

#         trend_df['NOISE'] = pd.concat(noise_series_list,axis='rows')

#         avg_series_list = []

#         for column in avg_df:
            
#             avg_series_list.append(avg_df[column])

#         trend_df['AVG'] = pd.concat(avg_series_list,axis='rows')
        
#         trend_df = trend_df.reset_index(drop=True)

#         noise_df = noise_df.transpose().reset_index(drop=True)
#         avg_df = avg_df.transpose().reset_index(drop=True)
#         time_df = time_df.transpose().reset_index(drop=True)

#         column_str_list = [f'{x:.3f}' for x in np.arange(0,0.61,0.075)]
#         column_float_list = [float(x) for x in column_str_list]
#         avg_df.columns = column_str_list

#         if model in ['LGBM','XGB','GB','ADA','RF','DEC']:
    
#             if model == 'DEC':

#                 model = 'DT'

#             model_group = 'Tree Based'
#             time_label = 'Time (ms)'
#             time = time_df.mean().mean() * 1000

#         elif model in ['ELAS','LASS','RIDGE']:

#             model_group = 'Linear'
#             time_label = 'Time (ms)'
#             time = time_df.mean().mean() * 1000

#         elif model in ['KNN','SVM','NN']:
    
#             model_group = 'Misc.'
#             time_label = 'Time (ms)'
#             time = time_df.mean().mean() * 1000

#         else:
    
#             model_group = 'Stack'
#             time_label = 'Time (ms)'
#             time = time_df.mean().mean() * 1000
#             print(time)

#         # plot #

#         ax.set_title(f'{model} ({model_group})',fontsize=22)

#         z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
#         p = np.poly1d(z)

#         ax.set_ylim(0,120)

#         slope = p.coefficients[0]
#         intercept = p.coefficients[1]

#         radar_df = pd.DataFrame({
#             'radar_col':['Sensitivity (F/g)','Error (F/g)',time_label],
#             model:[slope,intercept,time],
#             })

#         # theta has 5 different angles, and the first one repeated
#         theta = np.arange(len(radar_df) + 1) / float(len(radar_df)) * 2 * np.pi
#         # values has the 5 values from 'Col B', with the first element repeated
#         values = radar_df[model].values
#         values = np.append(values,values[0])

#         # draw the polygon and the mark the points for each angle/value combination
#         ax.plot(theta,values,marker=(3,0,model_count*(360/len(model_list))),label=model,color=colors_list[model_count],markersize=20,linewidth=3)
#         ax.set_xticks(theta[:-1],radar_df['radar_col'],color='grey',size=12)
#         ax.tick_params(pad=10) # to increase the distance of the labels to the plot
#         # fill the area of the polygon with green and some transparency
#         ax.fill(theta,values,alpha=0.15)

#         ax_label = 'abcdefghijklmnopqrstuvwxyz'
#         ax.text(x=-0.15,y=1.1,s=f'{ax_label[ax_count]})',transform=ax.transAxes,size=font_size_plot+5)

#     fig.tight_layout(rect=[0.05,0,0.95,1])

#     fig.savefig(f'{randomize_type}3.jpg',dpi=fig.dpi)

#     row_index = -1
#     col_index = -1

#     fig.clf()

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

row_index = -1
col_index = -1

for randomize_type in randomize_list:

    fig = plt.figure(figsize=(20,20))
    fig.suptitle('Summary Radar Plot',fontsize=font_size_plot*2)

    # fig.suptitle(f'Perturbed {randomize_type[0]+randomize_type[1:].lower()} Supercapacitor Dataset',fontsize=30)

    ax_tree = plt.subplot2grid((2,2),(0,0),fig=fig,projection='polar') # (rows,cols)
    ax_linear = plt.subplot2grid((2,2),(0,1),fig=fig,projection='polar') # (rows,cols)
    ax_misc = plt.subplot2grid((2,2),(1,0),fig=fig,projection='polar') # (rows,cols)
    ax_stack = plt.subplot2grid((2,2),(1,1),fig=fig,projection='polar') # (rows,cols)

    ax_list = [ax_tree,ax_linear,ax_misc,ax_stack]  
    ax_group_list = ['Tree-based','Linear','Misc.','Stack']

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    colors_list.append('#0f2734')
    colors_list.append('#ccff00')
    colors_list.append('#00ff00')
    patch_list = []

    for model_count,model in enumerate(model_list):

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
        time_df = time_df.transpose().reset_index(drop=True)

        column_str_list = [f'{x:.2f}' for x in np.arange(0,1.1,0.1)]
        column_float_list = [float(x) for x in column_str_list]
        avg_df.columns = column_str_list

        if model in ['LGBM','XGB','GB','ADA','RF','DEC']:
    
            if model == 'DEC':

                model = 'DT'

            model_group = 'Tree Based'
            time_label = 'Time (ms)'
            time = time_df.mean().mean() * 1000

            # plot #

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)

            ax_tree.set_ylim(0,120)

            slope = p.coefficients[0]
            intercept = p.coefficients[1]

            radar_df = pd.DataFrame({
                'radar_col':['Sensitivity (F/g)','Error (F/g)',time_label],
                model:[slope,intercept,time],
                })

            # theta has 5 different angles, and the first one repeated
            theta = np.arange(len(radar_df) + 1) / float(len(radar_df)) * 2 * np.pi
            # values has the 5 values from 'Col B', with the first element repeated
            values = radar_df[model].values
            values = np.append(values,values[0])

            # draw the polygon and the mark the points for each angle/value combination
            ax_tree.plot(theta,values,marker=(3,0,model_count*(360/len(model_list))),label=model,color=colors_list[model_count],markersize=20,linewidth=3)
            # fill the area of the polygon with green and some transparency
            ax_tree.fill(theta,values,alpha=0.15)
            ax_tree.grid(linewidth=5)

        elif model in ['ELAS','LASS','RIDGE']:

            model_group = 'Linear'
            time_label = 'Time (ms)'
            time = time_df.mean().mean() * 1000

            # plot #

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)

            ax_linear.set_ylim(0,120)

            slope = p.coefficients[0]
            intercept = p.coefficients[1]

            radar_df = pd.DataFrame({
                'radar_col':['Sensitivity (F/g)','Error (F/g)',time_label],
                model:[slope,intercept,time],
                })

            # theta has 5 different angles, and the first one repeated
            theta = np.arange(len(radar_df) + 1) / float(len(radar_df)) * 2 * np.pi
            # values has the 5 values from 'Col B', with the first element repeated
            values = radar_df[model].values
            values = np.append(values,values[0])

            # draw the polygon and the mark the points for each angle/value combination
            ax_linear.plot(theta,values,marker=(3,0,model_count*(360/len(model_list))),label=model,color=colors_list[model_count],markersize=20,linewidth=3)
            # fill the area of the polygon with green and some transparency
            ax_linear.fill(theta,values,alpha=0.15)
            ax_linear.grid(linewidth=5)

        elif model in ['KNN','SVM','NN']:
    
            model_group = 'Misc.'
            time_label = 'Time (ms)'
            time = time_df.mean().mean() * 1000

            # plot #

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)

            ax_misc.set_ylim(0,120)

            slope = p.coefficients[0]
            intercept = p.coefficients[1]

            radar_df = pd.DataFrame({
                'radar_col':['Sensitivity (F/g)','Error (F/g)',time_label],
                model:[slope,intercept,time],
                })

            # theta has 5 different angles, and the first one repeated
            theta = np.arange(len(radar_df) + 1) / float(len(radar_df)) * 2 * np.pi
            # values has the 5 values from 'Col B', with the first element repeated
            values = radar_df[model].values
            values = np.append(values,values[0])

            # draw the polygon and the mark the points for each angle/value combination
            ax_misc.plot(theta,values,marker=(3,0,model_count*(360/len(model_list))),label=model,color=colors_list[model_count],markersize=20,linewidth=3)
            # fill the area of the polygon with green and some transparency
            ax_misc.fill(theta,values,alpha=0.15)
            ax_misc.grid(linewidth=5)

        else:
    
            model_group = 'Stack'
            time_label = 'Time (ms)'
            time = time_df.mean().mean() * 1000

            # plot #

            z = np.polyfit(trend_df['NOISE'],trend_df['AVG'],1)
            p = np.poly1d(z)

            ax_stack.set_ylim(0,120)

            slope = p.coefficients[0]
            intercept = p.coefficients[1]

            radar_df = pd.DataFrame({
                'radar_col':['Sensitivity (F/g)','Error (F/g)',time_label],
                model:[slope,intercept,time],
                })

            # theta has 5 different angles, and the first one repeated
            theta = np.arange(len(radar_df) + 1) / float(len(radar_df)) * 2 * np.pi
            # values has the 5 values from 'Col B', with the first element repeated
            values = radar_df[model].values
            values = np.append(values,values[0])

            # draw the polygon and the mark the points for each angle/value combination
            ax_stack.plot(theta,values,marker=(3,0,model_count*(360/len(model_list))),label=model,color=colors_list[model_count],markersize=20,linewidth=3)
            # fill the area of the polygon with green and some transparency
            ax_stack.fill(theta,values,alpha=0.15)

            ax_stack.set_xticks(theta[:-1],radar_df['radar_col'],color='black')
            ax_stack.grid(linewidth=5)

    title_colors_list = ['red','blue','green','purple']

    for count,ax in enumerate(ax_list):

        ax.set_xticks(theta[:-1],radar_df['radar_col'],color='black')
        ax.tick_params(pad=20,labelsize=22)
        ax.set_rlabel_position(240)
        ax.set_title(ax_group_list[count],pad=40,color=title_colors_list[count])
        ax.legend(fontsize=20,bbox_to_anchor=(1.0,1.0),bbox_transform=ax.transAxes)
        # ax.set_theta_offset(270*np.pi/180)

        labels = []
        angle_theta = np.rad2deg(theta)
        for label,angle in zip(ax.get_xticklabels(),angle_theta):
            x,y = label.get_position()
            lab = ax.text(x,y,label.get_text(),transform=label.get_transform(),ha=label.get_ha(),va=label.get_va())
            lab.set_rotation(angle+270)
            labels.append(lab)
        ax.set_xticklabels([])

    fig.tight_layout()

    fig.savefig(f'{randomize_type}3.jpg',dpi=fig.dpi)

    row_index = -1
    col_index = -1

    plt.close()
