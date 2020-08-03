# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:25:15 2020

@author: gejianwen
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

__all__ = ["myScatter","myHeatMap"]

def myScatter(map_df, col1 = "x", col2 = "y", color = None, save_dir = "./temp/"):
    # plot it
    cm=plt.cm.inferno  # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    x = map_df[col1].values
    y = map_df[col2].values
    if not color:
        color = col1
    c = map_df[color].values
        
    plt.figure(figsize = (11,8))
    vmin = np.min(c)
    vmax = np.max(c)
    sc = plt.scatter(x, y, c=c, vmin=vmin, vmax=vmax, s=1, cmap=cm)
    plt.colorbar(sc)
    plt.xlabel(col1, fontsize = 20)
    plt.ylabel(col2,fontsize = 20)
    plt.title(col1+"-"+col2+"-"+color,fontsize = 20)
    plt.savefig(save_dir+"scatter_"+col1+"_"+col2+"_"+color+".png")
    return 0

def myHeatMap(harvest, col1, col2, save_dir = './temp/'):
    farmers = col1
    vegetables = col2
    fig,ax = plt.subplots(figsize = (9,9))
    im = ax.imshow(harvest)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(farmers)))
    ax.set_yticks(np.arange(len(vegetables)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(farmers)
    ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Correlationship")
    fig.tight_layout()
    plt.savefig( save_dir + "correlationship.png")
    plt.show()











