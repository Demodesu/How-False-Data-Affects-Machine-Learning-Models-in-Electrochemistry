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

pathlib.Path(f'{path}\\Plot').mkdir(parents=True,exist_ok=True)

df = pd.read_csv('Supercapacitor.csv')
cap_col = df.pop('CAP')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors_list = prop_cycle.by_key()['color']

fig = plt.figure(figsize=(20,11.25))

print(len(df.columns))

num_rows = 2
num_col = 4
row_index = 0
col_index = -1

for count,col in enumerate(df.columns):

    col_index += 1

    if col_index == num_col:

        col_index = 0

        row_index += 1

    print(row_index,col_index)

    ax = plt.subplot2grid((num_rows,num_col),(row_index,col_index),fig=fig)

    sns.scatterplot(x=df[col],y=cap_col,ax=ax,marker='D',color=colors_list[count],edgecolor='black')

    ax_label = 'abcdefghijklmnopqrstuvwxyz'
    ax.text(x=-0.15,y=1.1,s=f'{ax_label[count]})',transform=ax.transAxes,size=font_size_plot+5)

    ax.set_box_aspect(1)

fig.tight_layout()

fig.savefig('Plot\\Scatterplot.jpg',dpi=fig.dpi)