import time
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tushare as ts
ts.set_token('29eaf3bcac23df4c6d025de157ab2d53beead3391fbe6e83b4ebcb6c')
pro = ts.pro_api()

import mpl_finance as mpf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
import matplotlib.gridspec as gridspec
import matplotlib as mpl

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'SimHei'  # Chinese 
from mylab.stock.myfeature import getMa


__all__ = ["plotVol","plotKdj","plotMacd","plotCandle",
           "plotStock","plotStockByCode","displaySelect","plotGivenDay",
           ]


def plotGivenDay(daily_df,trade_date = "20200415", pre = 30, after = 30 , save_dir = "./temp/"):
    ix = daily_df.loc[daily_df["trade_date"].values == trade_date,:].index[0]
    if (ix-pre) >0:
        start_ix = (ix-pre)
    else:
        start_ix = 0
    if (ix+after) <len(daily_df):
        end_ix = (ix+after)
    else:
        end_ix = len(daily_df)

    display_df = daily_df.iloc[start_ix:end_ix, :]
    plotStock(display_df,focus_date = trade_date, SAVE = True,rolling = True, save_dir = save_dir)
    return display_df
    

def plotStock(daily_df, start_date = None, end_date = None, 
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
    plotKdj(daily_df, gs = gs)
    if SAVE:
        plt.savefig(save_dir+stock_code+"-"+ focus_date + ".png")
    return 0

def plotCandle(daily_df, stock_code = "" , focus_date = None, gs = None ):
    if not focus_date:
        focus_date = daily_df.trade_date.values[-1]
    # plot 
    if gs:
        ax1 = plt.subplot(gs[0:6,:])
    else:
        fig, ax1 = plt.subplots(figsize=(16,6))
    mpf.candlestick_ochl(
       ax=ax1,
       quotes=daily_df[['dates', 'open', 'close', 'high', 'low']].values,
       width=0.7,
       colorup='r',
       colordown='g',
       alpha=0.7)
    plt.grid(True)
    plt.xticks(rotation=40)
    plt.title(stock_code)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    date_tickers = daily_df["trade_date"].values[0::5]
    date_tickers = np.insert(date_tickers,0," ")
    ax1.set_xticklabels(date_tickers)
    # mean values
    for ma in ['lag_5', 'lag_10', 'lag_20', 'lag_30', 'lag_60', 'lag_120', 'lag_250']:
        plt.plot(daily_df['dates'], daily_df[ma], label = ma)
    plt.legend()
    # 分割线
    #    print(focus_date)
    x = daily_df.loc[daily_df["trade_date"] == focus_date, "dates"].values[0]
    plt.axvline(x,c="blue")#添加vertical直线
    y = daily_df.loc[daily_df["trade_date"] == focus_date, "close"].values[0]
    plt.axhline(y,c="blue",ls = "-.")#添加水平直线

    return 0

def plotVol(daily_df, gs = None):
    # plot vol
    if gs :        
        ax2 = plt.subplot(gs[7:9,:])
    else:
        fig, ax2 = plt.subplots(figsize=(16,3))
    daily_df['up'] = daily_df.apply(lambda row: 1 if row['close'] >= row['open'] else 0, axis=1)
    ax2.bar(daily_df.query('up == 1')['dates'], daily_df.query('up == 1')['vol'], color='r', alpha=0.7)
    ax2.bar(daily_df.query('up == 0')['dates'], daily_df.query('up == 0')['vol'], color='g', alpha=0.7)
    return 0


def plotMacd(daily_df, gs = None):
    if gs :        
        ax3 = plt.subplot(gs[9:11,:])
    else:
        fig, ax3 = plt.subplots(figsize=(16,3))
    plt.plot(daily_df.dif.values, "black")
    plt.plot(daily_df.dea.values, "orange")
    ax3.bar(daily_df.query('macd >= 0')['dates'], daily_df.query('macd >= 0')['macd'], color='r', alpha=0.7)
    ax3.bar(daily_df.query('macd < 0')['dates'], daily_df.query('macd < 0')['macd'], color='g', alpha=0.7)
    return 0

def plotKdj(daily_df, gs = None):
    if gs :        
        ax4 = plt.subplot(gs[11:,:])
    else:
        fig, ax4 = plt.subplots(figsize=(16,3))
    plt.plot(daily_df.K.values, "black",label = "K")
    plt.plot(daily_df.D.values, "orange",label = "D")
    plt.plot(daily_df.J.values, "purple",label = "J")
    plt.legend()
    return 0


def plotStockByCode(stock_code = "000001.SH", start_date = "20191210", end_date = "20200310"):  
    daily_df = pro.daily(ts_code = stock_code,start_date = start_date, end_date = end_date )
#    print(len(daily_df))
    plotStock(daily_df)
    
    return 0


def displaySelect(daily_df,ix ,trade_date = "", save_dir = "./temp/"):
    if (ix-60) >0:
        state_ix = (ix-60)
    else:
        state_ix = 0
    if (ix+20) <len(daily_df):
        end_ix = (ix+20)
    else:
        end_ix = len(daily_df)

    display_df = daily_df.iloc[state_ix:end_ix, :]
    plotStock(display_df,focus_date = trade_date, SAVE = True,rolling = True, save_dir = save_dir)
    return display_df



