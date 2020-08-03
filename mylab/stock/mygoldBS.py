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

__all__ = ["getGoldBS",
           "plotStockBS","plotGoldBS", "displaySelect",
           ]



def getGoldBS(df):
    df["AA"] = (df['open']+df['close']+df['high']+df['low'])/4.0  # AA:=(O+H+L+C)/4;
    df["BB"] = df["AA"].rolling(3).mean()                      # BB:=MA(AA,3);
    # IF(X,A,B)若X不为0则返回A，否则返回B。
    # SUM(X,N)，统计N周期中X的总和，N=0则从第一个有效值开始。
    # CC:=SUM(IF(AA>REF(AA,1),AA*VOL,0),4)/SUM(IF(AA<REF(AA,1),AA*VOL,0),4);
    df["AAxvol"] = df["AA"] * df["vol"]
    df["pre_AA"] = df["AA"] - df["AA"].diff(1)
    df["AA_rise"] = df["AA"]>df["pre_AA"]
    df["AA_rise"] = df["AA_rise"].astype(int)
    df["CC"] = (df["AAxvol"]*df["AA_rise"]).rolling(4).sum()/(df["AAxvol"]*(1-df["AA_rise"])).rolling(4).sum()
    # DD:=REF(100-(100/(1+CC)),1);
    df["DD"] = 100-(100/(1+df["CC"]))
    df["DD"] = df["DD"] - df["DD"].diff(1)
    # A1:=HHV(AA,15);  A2:=LLV(AA,15); A3:=A1-A2; A4:=EMA((AA-A2)/A3,2)*100;
    df["A1"] = df["AA"].rolling(15).max()      
    df["A2"] = df["AA"].rolling(15).min()      
    df["A3"] = df["A1"] - df["A2"]      
    df["A4"] = (df["AA"] - df["A2"])/df["A3"]      
    df["short"] = df["A4"].ewm(span = 2).mean()*100
    # B1:=HHV(AA,60); B2:=LLV(AA,60); B3:=B1-B2; B4:=EMA((AA-B2)/B3,2)*100;
    df["B1"] = df["AA"].rolling(60).max()      
    df["B2"] = df["AA"].rolling(60).min()      
    df["B3"] = df["B1"] - df["B2"]      
    df["B4"] = (df["AA"] - df["B2"])/df["B3"]      
    df["mid"] = df["B4"].ewm(span = 2).mean()*100
    # C1:=HHV(AA,240); C2:=LLV(AA,240); C3:=C1-C2; C4:=EMA((AA-C2)/C3,2)*100;
    df["C1"] = df["AA"].rolling(240).max()      
    df["C2"] = df["AA"].rolling(240).min()      
    df["C3"] = df["C1"] - df["C2"]      
    df["C4"] = (df["AA"] - df["C2"])/df["C3"]      
    df["long"] = df["C4"].ewm(span = 2).mean()*100
    # STICKLINE(短期>90 AND 中期>70,86,94,6,0),COLOR00FF00;
    df["sell_goldBS"] = np.all( [df["short"].values > 90,
                                df["mid"].values > 70,
                                ],
                               axis = 0)  
    df["buy_goldBS"] = np.all( [df["short"].values < 5,
                                df["mid"].values < 30,
                                ],
                               axis = 0)  
    
#    D1:=中期>REF(中期,1)AND 短期>REF(短期,1)AND 长期>REF(长期,1)AND 长期<8 AND 中期<10 AND 短期<15;
#    D2:=CROSS(短期,中期)AND 中期<8;
#    D3:=CROSS(短期,长期)AND 长期<8 AND 中期<20;
    df["D1"] =  np.all( [df["short"].diff(1).values > 0,
                         df["mid"].diff(1).values > 0,
                         df["long"].diff(1).values > 0,
                         df["short"].values < 15,
                         df["mid"].values < 10,
                         df["long"].values < 8,
                        ],
                       axis = 0)  
    df["cross_signal"] = df["short"].values > df['mid'].values
    df["cross_signal"] = df["cross_signal"].astype(int)
    df["cross_signal"] = df["cross_signal"].diff(1)  # CROSS(xx,yy)
    df["D2"] = np.all( [df["cross_signal"].values > 0,
                            df["mid"].values < 8,
                            ],
                           axis = 0)   
    df["cross_signal"] = df["short"].values > df['long'].values
    df["cross_signal"] = df["cross_signal"].astype(int)
    df["cross_signal"] = df["cross_signal"].diff(1)  # CROSS(xx,yy)
    df["D3"] = np.all( [df["cross_signal"].values > 0,
                            df["mid"].values < 20,
                            df["long"].values < 8,
                            ],
                           axis = 0)   
    temp1 =  np.any( [df["D1"].values,df["D2"].values,df["D3"].values], axis = 0) 
    # DRAWICON(FILTER((D1 OR D2 OR D3) AND DD<15 AND C>REF(C,1),5),5,1);
    df["boom"] = np.all( [df["DD"].values < 15,
                        df["close"].diff(1).values > 0,
                        temp1,
                        ],
                       axis = 0)   
    # DRAWICON(FILTER((中期<REF(中期,1)AND 短期<REF(短期,1)AND 长期<REF(长期,1)AND 中期>95 AND 短期>85)OR 
    # (长期>100 AND 中期>100 AND 短期>100),5),95,2);
    temp1 = np.all( [df["short"].diff(1).values < 0,
                    df["mid"].diff(1).values < 0,
                    df["long"].diff(1).values < 0,
                    df["short"].values > 85,
                    df["mid"].values > 95,
                    ],
                   axis = 0)
    temp2 = np.all( [df["short"].values > 99.9,
                    df["mid"].values > 99.9,
                    df["long"].values > 99.9,
                    ],
                   axis = 0)
    df["climax"] = np.any( [temp1, temp2], axis = 0)
    df.replace(np.nan, 0, inplace=True)
    df.replace(np.inf, 0, inplace=True)
    df["buy_goldBS"] = df["buy_goldBS"].astype(int)*20
    df["sell_goldBS"] = df["sell_goldBS"].astype(int)*20
    df["boom"] = df["boom"].astype(int)*40
    df["climax"] = df["climax"].astype(int)*40
    
    df["sell_short"] = df["short"].values > 90
    df["sell_short"] = df["sell_short"].astype(int)*10
    return df

def plotStockBS(daily_df, start_date = None, end_date = None, 
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
    plotGoldBS(daily_df, gs = gs)
    if SAVE:
        plt.savefig(save_dir+stock_code+"-"+ focus_date + ".png")
    return 0

def plotGoldBS(daily_df, gs = None):
    if gs :        
        ax4 = plt.subplot(gs[11:,:])
    else:
        fig, ax4 = plt.subplots(figsize=(16,3))
    
    plt.axhline(20,c="green",ls = "-")#添加水平直线
    plt.axhline(90,c="red",ls = "-")#添加水平直线
    
    plt.plot(daily_df.short.values, "red",label = "short")
    plt.plot(daily_df.mid.values, "black",label = "mid")
    plt.plot(daily_df.long.values, "orange",label = "long")
    
    ax4.bar(daily_df.query('buy_goldBS > 0')['dates'], daily_df.query('buy_goldBS > 0')['buy_goldBS'], color='r', alpha=0.7,label = "buy")
    ax4.bar(daily_df.query('sell_goldBS > 0')['dates'], daily_df.query('sell_goldBS > 0')['sell_goldBS'], color='green', alpha=0.7,label = "sell")
    ax4.bar(daily_df.query('boom > 0')['dates'], daily_df.query('boom > 0')['boom'], color='r', alpha=0.7,label = "boom")
    ax4.bar(daily_df.query('climax > 0')['dates'], daily_df.query('climax > 0')['climax'], color='green', alpha=0.7,label = "climax")

    plt.legend()
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
    plotStockBS(display_df,focus_date = trade_date, SAVE = True,rolling = True, save_dir = save_dir)
    return display_df











