# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:16:58 2020

@author: admin
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import mpl_finance as mpf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from mylab.stock.myfeature import getMacd
from mylab.stock.myfeature import getKdj
from mylab.stock.myfeature import getMa

from mylab.stock.myplot import plotVol
from mylab.stock.myplot import plotCandle
from mylab.stock.myplot import plotMacd
from mylab.stock.myplot import plotKdj

__all__ = ["get3Percent",
           "plotStock3Pct","plot3Percent", "displaySelect3Pct",
           ]



def get3Percent(df):
    var1 = (2*df['close']+df['high']+df['low'])/4.0
    var2 = df['low'].rolling(34, min_periods=34).min()
    var3 = df['high'].rolling(34, min_periods=34).max()
    df["var"] = (var1-var2)/(var3-var2)*100
    df["xx"]= df["var"].ewm(span=13).mean()
    df["xx1"] = (-df["xx"].diff(1)+df["xx"]*1.5)*2.0/3.0
    df["yy"] = df["xx1"].ewm(span=2).mean()
    
    df["cross_signal"] = df["xx"].values > df['yy'].values
    df["cross_signal"] = df["cross_signal"].astype(int)
    df["cross_signal"] = df["cross_signal"].diff(1)  # CROSS(xx,yy)
    df["buy_signal"] = np.all( [df["cross_signal"].values > 0,
                                df["xx"].values < 20,
                                ],
                               axis = 0)   # IF( CROSS(xx,yy) AND xx<20 ,30,0);
    
    sell_signal_reduce = np.all( [df["cross_signal"].values < 0,
                                df["xx"].values < 20,
                                ],
                               axis = 0)   # IF( CROSS(xx,yy) AND xx<20 ,30,0);
    df["cross_signal"] = df["xx"].values > 20
    df["cross_signal"] = df["cross_signal"].astype(int)
    df["cross_signal"] = df["cross_signal"].diff(1)   # CROSS(xx,20)
    sell_signal_3point = np.all( [df["cross_signal"].values > 0,
                                 df["xx"].values > df['yy'].values
                                ],
                                axis = 0)  # STICKLINE(CROSS(xx,20) AND yy<xx,xx+19,xx-10,4,0),colorred;
    df["sell_signal"] = np.any([sell_signal_reduce,sell_signal_3point],axis = 0)
    
    df.replace(np.nan, 0, inplace=True)
    df.replace(np.inf, 0, inplace=True)
    df["buy_signal"] = df["buy_signal"].astype(int)*30
    df["sell_signal"] = df["sell_signal"].astype(int)*30
    return df

def plotStock3Pct(daily_df, start_date = None, end_date = None, 
              focus_date = None, rolling = True,SAVE = False ,save_dir = "./picture/select_by_vol/"):
#    print(focus_date)
#    print(len(daily_df))
    stock_code = daily_df.ts_code.values[0]
    # add some ancillary column 
    daily_df["trade_date2"] = daily_df["trade_date"].copy()
    daily_df["trade_date2"]  = daily_df["trade_date2"].astype("str")
    daily_df["trade_date2"] = pd.to_datetime(daily_df["trade_date2"]).map(date2num)
    daily_df.sort_values(by="trade_date", ascending=True,inplace=True)
    daily_df["dates"] = np.arange(0,len(daily_df))
    if not focus_date:
        focus_date = daily_df.trade_date.values[-1]
    # whether cut
    if start_date:
        daily_df = daily_df.loc[daily_df["trade_date2"] >= start_date,:].reset_index(drop=True)
        daily_df['dates'] = np.arange(0, len(daily_df))
    if end_date:
        daily_df = daily_df.loc[daily_df["trade_date2"] <= end_date,:].reset_index(drop=True)
        daily_df['dates'] = np.arange(0, len(daily_df))
    # meanetwork value
    if rolling:
        daily_df = getMa(daily_df)
        
    # plot 
    fig = plt.subplots(figsize=(16,11))
    gs = gridspec.GridSpec(13, 1)
    plotCandle(daily_df, stock_code =stock_code, focus_date = focus_date,  gs = gs )
    plotVol(daily_df, gs = gs)
    plotMacd(daily_df, gs = gs)
    plot3Percent(daily_df, gs = gs)
    if SAVE:
        plt.savefig(save_dir+stock_code+"-"+ focus_date + ".png")
    return 0

def plot3Percent(daily_df, gs = None):
    if gs :        
        ax4 = plt.subplot(gs[11:,:])
    else:
        fig, ax4 = plt.subplots(figsize=(16,3))
    plt.plot(daily_df.xx.values, "black",label = "xx")
    plt.plot(daily_df.yy.values, "orange",label = "yy")
    plt.plot(daily_df.buy_signal.values, "red",label = "buy")
    plt.plot(daily_df.sell_signal.values, "green",label = "sell")
    plt.legend()
    return 0

def displaySelect3Pct(daily_df,ix ,trade_date = "", save_dir = "./temp/"):
    if (ix-60) >0:
        state_ix = (ix-60)
    else:
        state_ix = 0
    if (ix+20) <len(daily_df):
        end_ix = (ix+20)
    else:
        end_ix = len(daily_df)

    display_df = daily_df.iloc[state_ix:end_ix, :]
    plotStock3Pct(display_df,focus_date = trade_date, SAVE = True,rolling = True, save_dir = save_dir)
    return display_df











