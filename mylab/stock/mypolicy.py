# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:11:37 2020

@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

__all__ = ["initiateCols","saveInfo","calCriterion",
           "actionDay3Per","action3Per",
           "actionDayZT","actionZT",
           ]


def actionDay(stock_df,select_df,buy_day_ix,hold_period):
    stock_ix = select_df.pre_index.values[buy_day_ix] 
    act_day_ix = stock_ix+hold_period
    sell = True
    # reach max
    if stock_df.high.values[ act_day_ix ] > select_df.target_price.values[buy_day_ix]:
        select_df["sell_price"][buy_day_ix] = select_df.target_price.values[buy_day_ix]
        if stock_df.open.values[ act_day_ix ] > select_df.target_price.values[buy_day_ix]:
            select_df["sell_price"][buy_day_ix] = stock_df.open.values[ act_day_ix ]
        select_df["hold_period"][buy_day_ix] = hold_period
        select_df["sell_reason"][buy_day_ix] = "get target profit"
    # reach min
    elif stock_df.low.values[ act_day_ix] < select_df.minimal_price.values[buy_day_ix]:
        select_df["sell_price"][buy_day_ix] = stock_df.close.values[ act_day_ix]
        select_df["hold_period"][buy_day_ix] = hold_period
        select_df["sell_reason"][buy_day_ix] = "get minimal_price profit"
    # get sell signal
    elif stock_df.sell_signal.values[ act_day_ix] > 0:
        select_df["sell_price"][buy_day_ix] = stock_df.close.values[ act_day_ix ] 
        select_df["hold_period"][buy_day_ix] = hold_period
        select_df["sell_reason"][buy_day_ix] = "get sell signal"
    else:
        sell = False
        pass
    
    return select_df,sell
def action3Per(stock_df,select_df,buy_day_ix):
    stock_ix = select_df.pre_index.values[buy_day_ix]
    # day 1
    select_df, sell = actionDay(stock_df,select_df,buy_day_ix,1)
    if not sell:
        select_df, sell = actionDay(stock_df,select_df,buy_day_ix,2)
        if not sell:
            select_df, sell = actionDay(stock_df,select_df,buy_day_ix,3)
            if not sell:
                select_df, sell = actionDay(stock_df,select_df,buy_day_ix,4)
                if not sell:
                    select_df["sell_price"][buy_day_ix] = stock_df.close.values[ stock_ix + 4 ] 
                    select_df["hold_period"][buy_day_ix] = 4
                    select_df["sell_reason"][buy_day_ix] = "get max period"
    return select_df
    

def initiateCols(select_df):
    # ts_code	open	high	low 	close	pre_close	change	pct_chg	vol	amount	turnover_rate	volume_ratio
    select_df["postday_open"] = np.nan 
    select_df["postday_high"] = np.nan 
    select_df["postday_low"] = np.nan 
    select_df["postday_close"] = np.nan 
    select_df["postday_vol"] = np.nan 
    select_df["postday_pct_chg"] = np.nan 
    select_df["postday_amount"] = np.nan 
    select_df["postday_turnover_rate"] = np.nan 
    select_df["postday_volume_ratio"] = np.nan 
    
    select_df["preday_open"] = np.nan 
    select_df["preday_high"] = np.nan 
    select_df["preday_low"] = np.nan 
    select_df["preday_close"] = np.nan 
    select_df["preday_vol"] = np.nan 
    select_df["preday_pct_chg"] = np.nan 
    select_df["preday_amount"] = np.nan 
    select_df["preday_turnover_rate"] = np.nan 
    select_df["preday_volume_ratio"] = np.nan 
    
    select_df["buy_price"] = np.nan
    select_df["buy_amount"] = 1
    select_df["T1_price"] = np.nan
    select_df["T2_price"] = np.nan
    select_df["target_price"] = np.nan
    select_df["minimal_price"] = np.nan
    select_df["sell_price"] = np.nan
    select_df["sell2_price"] = np.nan
    select_df["hold_period"] = np.nan
    select_df["sell_reason"] = np.nan 
    return select_df

def saveInfo(select_df,stock_df,i):
    # get index
    select_date = select_df.loc[i,"trade_date"]
    stock_ix = stock_df.loc[stock_df["trade_date"].values == select_date,:].index[0]
    # store post day info
    select_df["postday_open"][i] = stock_df["open"][stock_ix+1]
    select_df["postday_high"][i] = stock_df["high"][stock_ix+1]
    select_df["postday_low"][i] = stock_df["low"][stock_ix+1]
    select_df["postday_close"][i] = stock_df["close"][stock_ix+1]
    select_df["postday_vol"][i] = stock_df["vol"][stock_ix+1]
    select_df["postday_pct_chg"][i] = stock_df["pct_chg"][stock_ix+1]
    select_df["postday_amount"][i] = stock_df["amount"][stock_ix+1]
    select_df["postday_turnover_rate"][i] = stock_df["trunover_rate"][stock_ix+1]
    select_df["postday_volume_ratio"][i] = stock_df["volume_ratio"][stock_ix+1]
    # store previous day data
    select_df["preday_open"][i] = stock_df["open"][stock_ix-1]
    select_df["preday_high"][i] = stock_df["high"][stock_ix-1]
    select_df["preday_low"][i] = stock_df["low"][stock_ix-1]
    select_df["preday_close"][i] = stock_df["close"][stock_ix-1]
    select_df["preday_vol"][i] = stock_df["vol"][stock_ix-1]
    select_df["preday_pct_chg"][i] = stock_df["pct_chg"][stock_ix-1]
    select_df["preday_amount"][i] = stock_df["amount"][stock_ix-1]
    select_df["preday_turnover_rate"][i] = stock_df["turnover_rate"][stock_ix-1]
    select_df["preday_volume_ratio"][i] = stock_df["volume_ratio"][stock_ix-1]
    # set policy parameters
    select_df["buy_price"][i] = stock_df["open"][stock_ix+1]
    select_df["buy_amount"][i] = 1
    select_df["target_price"][i] = round(select_df["buy_price"][i]*1.3, 2)  # 30% 止盈
    select_df["minimal_price"][i] = round(select_df["buy_price"][i]*0.95, 2) # 5% 止损
    
    return select_df

def calCriterion(select_df):
    # calculate some important inform after all buy-sell points are conducted
    select_df["earn_rate"] = (select_df["sell_price"]/select_df["buy_price"]-1)*100
    select_df["earn_rate_per_day"] = select_df["earn_rate"]/select_df["hold_period"]
    select_df["win"] = select_df["earn_rate"] > 0
    
    select_df["high_pct"] = (select_df["high"] /select_df["pre_close"] -1)*100
    select_df["postday_high_pct"] = (select_df["postday_high"] /select_df["close"] -1)*100
    select_df["postday_open_pct"] = (select_df["postday_open"] /select_df["close"] -1)*100
    select_df["postday_close_pct"] = (select_df["postday_close"] /select_df["close"] -1)*100
    
    return select_df

def actionDayZT(stock_df,select_df,buy_day_ix,hold_period):
    select_date = select_df.loc[buy_day_ix,"trade_date"]
    stock_ix = stock_df.loc[stock_df["trade_date"].values == select_date,:].index[0]
    act_day_ix = stock_ix+hold_period
    sell = True
    # no ZT
    if stock_df.close.values[ act_day_ix ] < round(stock_df.pre_close.values[ act_day_ix ]*1.1,2):
        select_df["hold_period"][buy_day_ix] = hold_period
        select_df["sell_reason"][buy_day_ix] = "not ZT"
        if hold_period == 1:
            select_df["sell_price"][buy_day_ix] = select_df.open.values[act_day_ix +1 ]
        else:
            select_df["sell_price"][buy_day_ix] = select_df.close.values[act_day_ix +1 ]
            
        if stock_df.open.values[ act_day_ix ] > select_df.target_price.values[buy_day_ix]:
            select_df["sell_price"][buy_day_ix] = stock_df.open.values[ act_day_ix ]
    else:
        sell = False
        pass
    
    return select_df,sell
def actionZT(stock_df,select_df,buy_day_ix):
    # get stock index
    select_date = select_df.loc[buy_day_ix,"trade_date"]
    stock_ix = stock_df.loc[stock_df["trade_date"].values == select_date,:].index[0]
    # if not ZT, buy it
    if stock_df.open.values[ stock_ix+1 ] < round(stock_df.close.values[ stock_ix ]*1.1,2):
        select_df["buy_price"][buy_day_ix] = stock_df.open.values[ stock_ix +1 ]
        # day 1
        select_df, sell = actionDayZT(stock_df,select_df,buy_day_ix,1)
        if not sell:
            select_df, sell = actionDayZT(stock_df,select_df,buy_day_ix,2)
            if not sell:
                select_df, sell = actionDayZT(stock_df,select_df,buy_day_ix,3)
                if not sell:
                    select_df, sell = actionDayZT(stock_df,select_df,buy_day_ix,4)
                    if not sell:
                        select_df["sell_price"][buy_day_ix] = stock_df.close.values[ stock_ix + 4 ] 
                        select_df["hold_period"][buy_day_ix] = 4
                        select_df["sell_reason"][buy_day_ix] = "get max period"
    return select_df