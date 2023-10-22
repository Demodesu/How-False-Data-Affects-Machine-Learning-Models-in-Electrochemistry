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

font_size_plot = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = font_size_plot

plt.gca().ticklabel_format(axis='both',style='plain',useOffset=False)

path = os.path.dirname(os.path.realpath(__file__))
os.chdir(f'{path}')

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

df = pd.read_csv('Supercapacitor.csv')
cap_col = df.pop('CAP')
df['CAP'] = cap_col

fig = plt.figure(figsize=(20,20))

correlation = df.select_dtypes('number').corr()
sns.heatmap(correlation,annot=True,cmap='seismic',vmax=1,vmin=-1)
plt.gca().set_aspect('equal')

fig.savefig('Plot\\Heatmap.jpg',dpi=fig.dpi)