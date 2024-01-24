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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

font_size_plot = 21
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

np.random.seed(0)
X = np.sort(5*np.random.rand(80,1),axis=0)
y_non_noisy = np.sin(X).ravel()
y_noisy = y_non_noisy + 0.5 * np.random.randn(80)
X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]
Y_test = np.sin(X_test).ravel()

fig = plt.figure(figsize=(20,11.25))

ax_standard_noise = plt.subplot2grid((1,2),(0,0),fig=fig)
ax_stack_noise = plt.subplot2grid((1,2),(0,1),fig=fig)

base_models = [
    ('dt',DecisionTreeRegressor(max_depth=8,random_state=0)),
    ('lr',LinearRegression()),
    ('rf',SVR())
]

stacking_regressor = StackingRegressor(estimators=base_models,final_estimator=LinearRegression())
decision_tree_regressor = DecisionTreeRegressor(max_depth=8,random_state=0)
linear_regressor = LinearRegression()
support_vector_machine_regressor = SVR()

plot_regressor_normal = [decision_tree_regressor,linear_regressor,support_vector_machine_regressor]
plot_regressor_normal_name = ['DT','LIN','SVM']
plot_regressor_stack = [stacking_regressor]
plot_regressor_stack_name = ['STACK']
ax_list = [ax_standard_noise,ax_stack_noise]
ax_name_list = ['Standalone Noisy Data','Stack Noisy Data']

for count,ax in enumerate(ax_list):

    if ax == ax_standard_noise:

        for model_count,model in enumerate(plot_regressor_normal):

            model.fit(X,y_noisy)
            prediction = model.predict(X_test)
            difference = abs(prediction - Y_test).mean()

            if model_count == 0:

                ax.scatter(X,y_noisy,edgecolor='black',c='orange',label='Noisy data',s=70)
                ax.plot(X_test,prediction,label=f'{plot_regressor_normal_name[model_count]} MAE ({difference:.2f})',linewidth=3)        

            else:

                ax.scatter(X,y_noisy,edgecolor='black',c='orange',s=70)
                ax.plot(X_test,prediction,label=f'{plot_regressor_normal_name[model_count]} MAE ({difference:.2f})',linewidth=3)        

    if ax == ax_stack_noise:

        for model_count,model in enumerate(plot_regressor_stack):

            model.fit(X,y_noisy)
            prediction = model.predict(X_test)
            difference = abs(prediction - Y_test).mean()

            ax.scatter(X,y_noisy,edgecolor='black',c='orange',label='Noisy data',s=70)
            ax.plot(X_test,prediction,label=f'{plot_regressor_stack_name[model_count]} MAE ({difference:.2f})',linewidth=3)   

    ax.set_title(ax_name_list[count])
    ax.set_box_aspect(1)
    ax.plot(X,y_non_noisy,c='black',linewidth=3,label='True function')
    ax.legend(loc='lower left')

    ax_label = 'abcdefghijklmnopqrstuvwxyz'
    ax.text(x=-0.15,y=1.1,s=f'{ax_label[count]})',transform=ax.transAxes,size=font_size_plot+10)

fig.tight_layout()

fig.savefig('Plot\\Stack.jpg',dpi=fig.dpi)