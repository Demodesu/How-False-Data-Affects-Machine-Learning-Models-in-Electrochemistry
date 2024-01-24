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

font_size_plot = 22
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

fig = plt.figure(figsize=(20,11.25))

ax_noise = plt.subplot2grid((1,2),(0,0),fig=fig)
ax_non_noise = plt.subplot2grid((1,2),(0,1),fig=fig)

np.random.seed(0)
X = np.sort(5*np.random.rand(80,1),axis=0)
y_non_noisy = np.sin(X).ravel()
y_noisy = y_non_noisy + 0.5 * np.random.randn(80)

regressor_non_noisy = LinearRegression()
regressor_noisy = LinearRegression()

regressor_non_noisy.fit(X,y_non_noisy)
regressor_noisy.fit(X,y_noisy)

X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]

y_non_noisy_pred = regressor_non_noisy.predict(X_test)
y_noisy_pred = regressor_noisy.predict(X_test)

ax_noise.scatter(X,y_noisy,edgecolor='black',c='orange',label='Noisy data',s=100)
ax_noise.plot(X_test,y_noisy_pred,color='blue',label='Noisy model',linewidth=3)
ax_non_noise.scatter(X,y_non_noisy,edgecolor='black',c='pink',label='Non-noisy data',s=100)
ax_non_noise.plot(X_test,y_non_noisy_pred,color='red',label='Non-noisy model',linewidth=3)

ax_noise.set_box_aspect(1)
ax_non_noise.set_box_aspect(1)

ax_noise.set_xlabel('Data')
ax_noise.set_ylabel('Target')
ax_noise.set_title('Linear Noisy Data')
ax_noise.legend(loc='lower left')

ax_non_noise.set_xlabel('Data')
ax_non_noise.set_ylabel('Target')
ax_non_noise.set_title('Linear Normal Data')
ax_non_noise.legend(loc='lower left')

ax_label = 'abcdefghijklmnopqrstuvwxyz'
ax_list = [ax_noise,ax_non_noise]

for i in range(2):

    ax_list[i].text(x=-0.15,y=1.1,s=f'{ax_label[i]})',transform=ax_list[i].transAxes,size=font_size_plot+5)

fig.tight_layout(rect=[0.05,0.05,0.95,0.95])

fig.savefig('Plot\\LIN.jpg',dpi=fig.dpi)