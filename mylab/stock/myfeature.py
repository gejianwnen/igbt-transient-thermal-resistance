
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import date2num

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'SimHei'  # Chinese 


__all__ = ["getMa","getKdj","getMacd","getPre",
           "extractFeature",
           "resetIndex"
           ]


def resetIndex(daily_df):
        # reset ascending
    daily_df["trade_date_stamp"] = daily_df["trade_date"].copy()
    daily_df["trade_date_stamp"] = pd.to_datetime(daily_df["trade_date_stamp"]).map(date2num)
    daily_df.sort_values(by="trade_date_stamp", ascending=True,inplace=True)
    daily_df.reset_index(drop=True,inplace=True)
    return daily_df

def getMa(daily_df):
    daily_df['lag_5'] = daily_df.close.rolling(5).mean()
    daily_df['lag_10'] = daily_df.close.rolling(10).mean()
    daily_df['lag_20'] = daily_df.close.rolling(20).mean()
    daily_df['lag_30'] = daily_df.close.rolling(30).mean()
    daily_df['lag_60'] = daily_df.close.rolling(60).mean()
    daily_df['lag_120'] = daily_df.close.rolling(120).mean()
    daily_df['lag_250'] = daily_df.close.rolling(250).mean()
    
    return daily_df


def getMacd(data,short=None,long=None,mid=None):
    if not short:
        short=12
    if not long:
        long=26
    if not mid:
        mid=9
    #计算短期的ema，使用pandas的ewm得到指数加权的方法，mean方法指定数据用于平均
    data['sema']=pd.Series(data['close']).ewm(span=short).mean()
    #计算长期的ema，方式同上
    data['lema']=pd.Series(data['close']).ewm(span=long).mean()
    #填充为na的数据
    data.fillna(0,inplace=True)
    #计算dif，加入新列data_dif
    data['dif']=data['sema']-data['lema']
    #计算dea
    data['dea']=pd.Series(data['dif']).ewm(span=mid).mean()
    #计算macd
    data['macd']=2*(data['dif']-data['dea'])
    # 计算 signal
    data["macd_signal"] = data["dif"].values > data['dea'].values
    data["macd_signal"] = data["macd_signal"].astype(int)
    data["macd_signal"] = data["macd_signal"].diff()
    return data

def getKdj(df):
    low_list = df['low'].rolling(9, min_periods=9).min()
    low_list.fillna(value = df['low'].expanding().min(), inplace = True)
    high_list = df['high'].rolling(9, min_periods=9).max()
    high_list.fillna(value = df['high'].expanding().max(), inplace = True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100

    df['K'] = pd.DataFrame(rsv).ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    df["kdj_signal"] = df["K"].values > df['D'].values
    df["kdj_signal"] = df["kdj_signal"].astype(int)
    df["kdj_signal"] = df["kdj_signal"].diff()
    
    return df

def getPre(daily_df):
        # store previous data values
    daily_df['vol_pre1'] = np.insert(daily_df['vol'].values[:-1],0,np.NaN)
    daily_df['vol_pre2'] = np.insert(daily_df['vol_pre1'].values[:-1],0,np.NaN)
    daily_df['vol_pre3'] = np.insert(daily_df['vol_pre2'].values[:-1],0,np.NaN)
    daily_df['vol_pre4'] = np.insert(daily_df['vol_pre3'].values[:-1],0,np.NaN)
    daily_df['vol_pre5'] = np.insert(daily_df['vol_pre4'].values[:-1],0,np.NaN)
    daily_df['vol_add'] = daily_df['vol']/daily_df['vol_pre1']
    daily_df['vol_add_pre1'] = daily_df['vol_pre1']/daily_df['vol_pre2']
    daily_df['vol_add_pre2'] = daily_df['vol_pre2']/daily_df['vol_pre3']
    daily_df['vol_add_pre3'] = daily_df['vol_pre3']/daily_df['vol_pre4']
    daily_df['vol_add_pre4'] = daily_df['vol_pre4']/daily_df['vol_pre5']
    
    daily_df['open_pre1'] = np.insert(daily_df['open'].values[:-1],0,np.NaN)
    daily_df['open_pre2'] = np.insert(daily_df['open_pre1'].values[:-1],0,np.NaN)
    daily_df['open_pre3'] = np.insert(daily_df['open_pre2'].values[:-1],0,np.NaN)
    daily_df['open_pre4'] = np.insert(daily_df['open_pre3'].values[:-1],0,np.NaN)
    daily_df['open_pre5'] = np.insert(daily_df['open_pre4'].values[:-1],0,np.NaN)
    
    daily_df['close_pre1'] = np.insert(daily_df['close'].values[:-1],0,np.NaN)
    daily_df['close_pre2'] = np.insert(daily_df['close_pre1'].values[:-1],0,np.NaN)
    daily_df['close_pre3'] = np.insert(daily_df['close_pre2'].values[:-1],0,np.NaN)
    daily_df['close_pre4'] = np.insert(daily_df['close_pre3'].values[:-1],0,np.NaN)
    daily_df['close_pre5'] = np.insert(daily_df['close_pre4'].values[:-1],0,np.NaN)

    daily_df['change_pre1'] = np.insert(daily_df['pct_chg'].values[:-1],0,np.NaN)
    daily_df['change_pre2'] = np.insert(daily_df['change_pre1'].values[:-1],0,np.NaN)
    daily_df['change_pre3'] = np.insert(daily_df['change_pre2'].values[:-1],0,np.NaN)
    daily_df['change_pre4'] = np.insert(daily_df['change_pre3'].values[:-1],0,np.NaN)
    daily_df['change_pre5'] = np.insert(daily_df['change_pre4'].values[:-1],0,np.NaN)
    
    daily_df["change_day"] = (daily_df["close"]-daily_df['open'])/daily_df["close"]*100
    daily_df["change_pre1_day"] = (daily_df["close_pre1"]-daily_df['open_pre1'])/daily_df["open_pre1"]*100
    daily_df["change_pre2_day"] = (daily_df["close_pre2"]-daily_df['open_pre2'])/daily_df["open_pre2"]*100
    daily_df["change_pre3_day"] = (daily_df["close_pre3"]-daily_df['open_pre3'])/daily_df["open_pre3"]*100
    daily_df["change_pre4_day"] = (daily_df["close_pre4"]-daily_df['open_pre4'])/daily_df["open_pre4"]*100
    daily_df["change_pre5_day"] = (daily_df["close_pre5"]-daily_df['open_pre5'])/daily_df["open_pre5"]*100

    daily_df["vol_mean_pre5"] = (daily_df['vol_pre1'] +daily_df['vol_pre2'] +daily_df['vol_pre3'] +daily_df['vol_pre4'] +daily_df['vol_pre5'])/5.0
    daily_df["quantity_relative_ratio"] = daily_df["vol"]/daily_df["vol_mean_pre5"]

    daily_df["vol_scale"] = daily_df["vol"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre1_scale"] = daily_df["vol_pre1"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre2_scale"] = daily_df["vol_pre2"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre3_scale"] = daily_df["vol_pre3"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre4_scale"] = daily_df["vol_pre4"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre5_scale"] = daily_df["vol_pre5"]/daily_df["vol_mean_pre5"]
    daily_df["vol_pre5_std"] = daily_df[["vol_pre1_scale","vol_pre2_scale","vol_pre3_scale","vol_pre4_scale","vol_pre5_scale"]].std(axis = 1)
    daily_df["vol_pre5_max"] = daily_df[["vol_pre1_scale","vol_pre2_scale","vol_pre3_scale","vol_pre4_scale","vol_pre5_scale"]].max(axis = 1)
    daily_df["vol_pre5_min"] = daily_df[["vol_pre1_scale","vol_pre2_scale","vol_pre3_scale","vol_pre4_scale","vol_pre5_scale"]].min(axis = 1)
    daily_df["vol_pre5_amp"] = daily_df["vol_pre5_max"] - daily_df["vol_pre5_min"]
    daily_df["close_pre5_mean"] = daily_df[["close_pre1","close_pre2","close_pre3","close_pre4","close_pre5"]].mean(axis = 1)
    daily_df["close_pre5_std"] = daily_df[["close_pre1","close_pre2","close_pre3","close_pre4","close_pre5"]].std(axis = 1)
    daily_df["close_pre5_max"] = daily_df[["close_pre1","close_pre2","close_pre3","close_pre4","close_pre5"]].max(axis = 1)
    daily_df["close_pre5_min"] = daily_df[["close_pre1","close_pre2","close_pre3","close_pre4","close_pre5"]].min(axis = 1)
    daily_df["change_pre5_mean"] = daily_df[["change_pre1","change_pre2","change_pre3","change_pre4","change_pre5"]].mean(axis = 1)
    daily_df["change_pre5_std"] = daily_df[["change_pre1","change_pre2","change_pre3","change_pre4","change_pre5"]].std(axis = 1)
    daily_df["change_pre5_max"] = daily_df[["change_pre1","change_pre2","change_pre3","change_pre4","change_pre5"]].max(axis = 1)
    daily_df["change_pre5_min"] = daily_df[["change_pre1","change_pre2","change_pre3","change_pre4","change_pre5"]].min(axis = 1)
    
    return daily_df


def extractFeature(daily_df):
    # add some ancillary column and sort by time
    daily_df = resetIndex(daily_df)
    # get macd
    daily_df = getMacd(daily_df)
    daily_df = getKdj(daily_df)
    # meanetwork value
    daily_df = getMa(daily_df)
    daily_df = getPre(daily_df)
    
    return daily_df



