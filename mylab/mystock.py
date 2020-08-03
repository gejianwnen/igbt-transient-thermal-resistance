import time
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tushare as ts
ts.set_token('3e74f4436e8b7fc7554e9eb495ffaa85aaceeff4ae72d845ac662832')
pro = ts.pro_api()

import mpl_finance as mpf
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pylab import date2num
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
sns.set()
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'SimHei'  # Chinese 

def getStockBasic(LOCAL = True):
    if LOCAL:
        pool_df = pd.read_csv("./data/stock/stock_basic_info.csv")
    else:
        pool_df = pro.stock_basic()
    return pool_df

def getIndexBasic(LOCAL = True, market = "SZSE" ):
    if LOCAL:
        pool_df = pd.read_csv("./data/index/index_basic_info_"+market+".csv")
    else:
        pool_df = pro.index_basic(market= market)
    return pool_df

def getIndexDaily(stock_code , start_date = "20100101", end_date = "20200314", LOCAL = True, market = "SZSE" ):
    dir_file = "./data/index/"+market+"/daily/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.index_daily(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getIndexWeekly(stock_code, start_date = "20100101", end_date = "20200314", LOCAL = True, market = "SZSE" ):
    dir_file = "./data/index/"+market+"/weekly/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.index_weekly(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getIndexMonthly(stock_code, start_date = "20100101", end_date = "20200314", LOCAL = True, market = "SZSE" ):
    dir_file = "./data/index/"+market+"/monthly/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.index_monthly(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df
    return daily_df

def getStockDaily(stock_code, start_date = "20100101", end_date = "20200314", LOCAL = True ):
    dir_file = "./data/stock/daily/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.daily(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getStockWeekly(stock_code, start_date = "20100101", end_date = "20200314", LOCAL = True ):
    dir_file = "./data/stock/weekly/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.daily(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getStockMonthly(stock_code, start_date = "20100101", end_date = "20200314", LOCAL = True ):
    dir_file = "./data/stock/monthly/"
    if LOCAL:
        daily_df = readData(dir_file, stock_code, start_date , end_date )
    else:
        daily_df = pro.daily(ts_code = stock_code,start_date = start_date, end_date = end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def resetIndex(daily_df):
        # reset ascending
    daily_df["trade_date_stamp"] = daily_df["trade_date"].copy()
    daily_df["trade_date_stamp"] = pd.to_datetime(daily_df["trade_date_stamp"]).map(date2num)
    daily_df.sort_values(by="trade_date_stamp", ascending=True,inplace=True)
    daily_df.reset_index(drop=True,inplace=True)
    return daily_df

def readData(dir_file, stock_code, start_date = "20100101", end_date = "20200314" ):
    for file_dir , _ , files in os.walk(dir_file):
        for i,file_name in enumerate(files):
            if file_name[:9] == stock_code:
                daily_df = pd.read_csv(file_dir+file_name)
                daily_df["trade_date"] = daily_df["trade_date"].astype("str")
                daily_df = daily_df.loc[daily_df["trade_date"] >= start_date,:].reset_index(drop=True)
                daily_df = daily_df.loc[daily_df["trade_date"] <= end_date,:].reset_index(drop=True)
                break
    return daily_df

def mergeDailyWeeklyMonthly(daily_df,weekly_df,monthly_df):
    weekly_df.drop(["ts_code", "trade_date_stamp"],axis = 1, inplace = True)
    cols = [i+'_weekly' for i in weekly_df.columns ]
    weekly_df.columns = cols
    weekly_df.rename(columns = {"trade_date_weekly":"trade_date"}, inplace = True)
    all_df = pd.merge(daily_df, weekly_df, how= "left"  ,on= "trade_date")
    monthly_df.drop(["ts_code", "trade_date_stamp"],axis = 1, inplace = True)
    cols = [i+'_monthly' for i in monthly_df.columns ]
    monthly_df.columns = cols
    monthly_df.rename(columns = {"trade_date_monthly":"trade_date"}, inplace = True)
    all_df = pd.merge(all_df, monthly_df, how= "left"  ,on= "trade_date")
    
    all_df.fillna(method= "ffill", inplace=True)
    return all_df

def mergeStockIndex(stock_df, df):
    index_df = df.copy(deep = True)
    index_df.drop(["ts_code", "trade_date_stamp"],axis = 1, inplace = True)
    cols = [i+'_index' for i in index_df.columns.values ]
    index_df.columns = cols
    index_df.rename(columns = {"trade_date_index":"trade_date"}, inplace = True)
    all_df = pd.merge(left = stock_df, right = index_df, how= "left"  ,on= "trade_date")
    return all_df

def getStock(stock_code,start_date, end_date  , LOCAL = True):
    daily_df = getStockDaily(stock_code,start_date, end_date  , LOCAL = True)
    weekly_df = getStockWeekly(stock_code,start_date , end_date  , LOCAL = True)
    monthly_df = getStockMonthly(stock_code,start_date , end_date  , LOCAL = True)
    # KDJ and MACD
    daily_df = getKdj(daily_df)
    daily_df = getMacd(daily_df)
    weekly_df = getKdj(weekly_df)
    weekly_df = getMacd(weekly_df)
    monthly_df = getKdj(monthly_df)
    monthly_df = getMacd(monthly_df)
    # merge
    all_df = mergeDailyWeeklyMonthly(daily_df,weekly_df,monthly_df)
    return all_df

def getIndex(stock_code,start_date, end_date  , LOCAL = True):
    daily_df = getIndexDaily(stock_code,start_date, end_date  , LOCAL = True)
    weekly_df = getIndexWeekly(stock_code,start_date , end_date  , LOCAL = True)
    monthly_df = getIndexMonthly(stock_code,start_date , end_date  , LOCAL = True)
    # KDJ
    daily_df = getKdj(daily_df)
    daily_df = getMacd(daily_df)
    weekly_df = getKdj(weekly_df)
    weekly_df = getMacd(weekly_df)
    monthly_df = getKdj(monthly_df)
    monthly_df = getMacd(monthly_df)
    # merge
    all_df = mergeDailyWeeklyMonthly(daily_df,weekly_df,monthly_df)
    return all_df


def plotStock(daily_df, start_date = None, end_date = None, 
              focus_date = None, rolling = True,SAVE = False ,save_dir = "./picture/select_by_vol/"):
#    print(focus_date)
#    print(len(daily_df))
    stock_code = daily_df.ts_code.values[0]
    # add some ancillary column 
    daily_df["trade_date2"] = daily_df["trade_date"].copy()
    daily_df["trade_date2"]  = daily_df["trade_date2"].astype("str")
    daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"]).map(date2num)
    daily_df.sort_values(by="trade_date2", ascending=True,inplace=True)
    daily_df["dates"] = np.arange(0,len(daily_df))
    if not focus_date:
        focus_date = daily_df.trade_date2.values[-1]
    # whether cut
    if start_date:
        daily_df = daily_df.loc[daily_df["trade_date"] >= start_date,:].reset_index(drop=True)
        daily_df['dates'] = np.arange(0, len(daily_df))
    if end_date:
        daily_df = daily_df.loc[daily_df["trade_date"] <= end_date,:].reset_index(drop=True)
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
        focus_date = daily_df.trade_date2.values[-1]
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
    date_tickers = daily_df["trade_date2"].values[0::5]
    date_tickers = np.insert(date_tickers,0," ")
    ax1.set_xticklabels(date_tickers)
    # mean values
    for ma in ['lag_5', 'lag_10', 'lag_20', 'lag_30', 'lag_60', 'lag_120', 'lag_250']:
        plt.plot(daily_df['dates'], daily_df[ma], label = ma)
    plt.legend()
    # 分割线
    #    print(focus_date)
    x = daily_df.loc[daily_df["trade_date2"] == focus_date, "dates"].values[0]
    plt.axvline(x,c="blue")#添加vertical直线
    y = daily_df.loc[daily_df["trade_date2"] == focus_date, "close"].values[0]
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

def getMa(daily_df):
    daily_df['lag_5'] = daily_df.close.rolling(5).mean()
    daily_df['lag_10'] = daily_df.close.rolling(10).mean()
    daily_df['lag_20'] = daily_df.close.rolling(20).mean()
    daily_df['lag_30'] = daily_df.close.rolling(30).mean()
    daily_df['lag_60'] = daily_df.close.rolling(60).mean()
    daily_df['lag_120'] = daily_df.close.rolling(120).mean()
    daily_df['lag_250'] = daily_df.close.rolling(250).mean()
    
    return daily_df

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

def selectStock1(daily_df):
    # 条件选股
    # 增量大于 1.5
    # 前几天的交易量变化不大 std < 
    # 价格变动应该也不大 std <
    # 这两天股票是涨的，并且涨幅在 0.1 到 8.0 之间
    select_df = daily_df.copy(deep = True)
    select_df = select_df.loc[select_df.vol_add.values> 1.2 ,:]   # 当日量比
    select_df = select_df.loc[select_df.vol_add.values< 1.8 ,:]
    select_df = select_df.loc[select_df.vol_add_pre1.values> 1.1 ,:]   # 前一天日量比
    select_df = select_df.loc[select_df.vol_pre5_std.values < 0.3 ,:]   #五日放量波动
    select_df = select_df.loc[select_df.change_pre5_std.values < 1 ,:]  # 五日价格波动
    select_df = select_df.loc[select_df.pct_chg.values < 6 ,:] # 当日股价增加百分比
    select_df = select_df.loc[select_df.pct_chg.values > 1 ,:]
    select_df = select_df.loc[select_df.change_pre1.values < 5 ,:]  # 前一日股价增加百分比
    select_df = select_df.loc[select_df.change_pre1.values > 0 ,:]
    select_df = select_df.loc[select_df.close_pre5_mean.values < 10,:]  # 前五日均价
    
#    print(select_df.shape)
    select_df.head()
    
    return select_df

def selectStock2(daily_df):
    # 条件选股
    # 增量大于 1.5
    # 前几天的交易量变化不大 std < 
    # 价格变动应该也不大 std <
    # 这两天股票是涨的，并且涨幅在 0.1 到 8.0 之间
    select_df = daily_df.copy(deep = True)
    select_df = select_df.loc[select_df.vol_add.values> 1.2 ,:]   # 当日量比
    select_df = select_df.loc[select_df.vol_add.values< 1.8 ,:]
    select_df = select_df.loc[select_df.vol_add_pre1.values> 1.1 ,:]   # 昨天日量比
    select_df = select_df.loc[select_df.vol_add_pre1.values> 1.5 ,:]   # 昨天日量比
    select_df = select_df.loc[select_df.vol_add_pre2.values> 1.1 ,:]   # 前天日量比
    select_df = select_df.loc[select_df.vol_add_pre2.values> 1.3 ,:]   # 前天日量比
    

    select_df = select_df.loc[select_df.pct_chg.values < 8 ,:] # 当日股价增加百分比
    select_df = select_df.loc[select_df.pct_chg.values > 2 ,:]
    select_df = select_df.loc[select_df.change_pre1.values < 6 ,:]  # 昨日股价增加百分比
    select_df = select_df.loc[select_df.change_pre1.values > 1 ,:]
    select_df = select_df.loc[select_df.change_pre2.values < 5 ,:]  # 前日股价增加百分比
    select_df = select_df.loc[select_df.change_pre2.values > 0.5 ,:]
    
    select_df = select_df.loc[select_df.change_pre1_day.values > 0.5 ,:]
    select_df = select_df.loc[select_df.change_pre1_day.values < 5 ,:]
    select_df = select_df.loc[select_df.change_day.values > 1 ,:]
    select_df = select_df.loc[select_df.change_day.values < 6 ,:]
    
    select_df = select_df.loc[select_df.close_pre5_mean.values < 15,:]  # 前五日均价
    
#    print(select_df.shape)
    select_df.head()
    
    return select_df

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



