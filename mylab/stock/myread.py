# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tushare as ts
ts.set_token('29eaf3bcac23df4c6d025de157ab2d53beead3391fbe6e83b4ebcb6c')
pro = ts.pro_api()

from matplotlib.pylab import date2num
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'SimHei'  # Chinese 

from mylab.stock.myfeature import getKdj
from mylab.stock.myfeature import getMacd


__all__ = ["getIndexBasic","getIndexDaily", "getIndexWeekly","getIndexMonthly","getIndex",
           "getStockBasic","getStockDaily","getStockWeekly","getStockMonthly","getStock",
           "getIndustryBasic","getIndustryDaily","getIndustryWeekly","getIndustryMonthly","getIndustry",
           "resetIndex","readData",
           "mergeDailyWeeklyMonthly","mergeWeeklyMonthly","mergeStockIndex",
           "deleteSTKC","deleteNew",
           "myMerge",
           ]

def myMerge(df1,df2,on = [], how = "left" ):
    cols = [i for i in df2.columns.values if i in df1.columns.values] # in df1 and df2
    cols = [i for i in cols if i not in on ]   # not in on
    df2 = df2.drop(cols, axis = 1 )
    df =  pd.merge( df1, df2, on = on , how = how )  
    return df

def deleteSTKC(pool_df):
    pool_df["name1"] = [i[0] for i in  pool_df["name"].values]
    pool_df["code1"] = [i[0] for i in  pool_df["ts_code"].values]
    pool_df["code3"] = [i[0:3] for i in  pool_df["ts_code"].values]
    pool_df = pool_df.loc[pool_df["name1"] != "*", :]
    pool_df = pool_df.loc[pool_df["name1"] != "S", :]
    pool_df = pool_df.loc[pool_df["code1"] != "3", :]
    pool_df = pool_df.loc[pool_df["code3"] != "688", :]
    pool_df = pool_df.drop(["name1","code1","code3"], axis = 1)
    pool_df = pool_df.reset_index(drop = True)
    return pool_df

def deleteNew(pool_df, list_data = "20190101"):
    pool_df = pool_df.loc[pool_df.list_date.values < list_data,:]
    pool_df = pool_df.reset_index(drop = True)
    return pool_df    

def getStockBasic(LOCAL = True, noSTKC = True, list_data = "20190101"):
    if LOCAL:
        pool_df = pd.read_csv("./data/stock/stock_basic_info.csv")
        pool_df["list_date"] = pool_df["list_date"].astype("str")
    else:
        fields='ts_code,symbol,name,area,industry,list_date,market,list_status,delist_date,exchange'
        pool_df = pro.stock_basic(list_status='L', fields=fields)
    if noSTKC:
        pool_df = deleteSTKC(pool_df)
    if list_data:
        pool_df =  deleteNew(pool_df, list_data )
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

def getIndustryBasic( ):
    pool_df = pd.read_csv("./data/industry/all_industry_basic_info.csv")
    return pool_df

def getIndustryDaily(stock_code , start_date = "20100101", end_date = "20200314" ):
    dir_file = "./data/industry/daily/"
    daily_df = readData(dir_file, stock_code, start_date , end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getIndustryWeekly(stock_code, start_date = "20100101", end_date = "20200314" ):
    dir_file = "./data/industry/weekly/"
    daily_df = readData(dir_file, stock_code, start_date , end_date )
    daily_df = resetIndex(daily_df)
    return daily_df

def getIndustryMonthly(stock_code, start_date = "20100101", end_date = "20200314" ):
    dir_file = "./data/industry/monthly/"
    daily_df = readData(dir_file, stock_code, start_date , end_date )
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

def mergeWeeklyMonthly(weekly_df,monthly_df):
    cols = [i+'_weekly' for i in weekly_df.columns ]
    weekly_df.columns = cols
    col_dic = {"trade_date_weekly":"trade_date","ts_code_weekly":"ts_code","trade_date_stamp_weekly":"trade_date_stamp"}
    weekly_df.rename(columns = col_dic, inplace = True)

    monthly_df.drop(["ts_code", "trade_date_stamp"],axis = 1, inplace = True)
    cols = [i+'_monthly' for i in monthly_df.columns ]
    monthly_df.columns = cols
    monthly_df.rename(columns = {"trade_date_monthly":"trade_date"}, inplace = True)
    
    all_df = pd.merge(weekly_df, monthly_df, how= "outer"  ,on= "trade_date")
    
    all_df.fillna(method= "ffill", inplace=True)
    return all_df

def mergeStockIndex(stock_df, df):
    index_df = df.copy(deep = True)
    index_df.drop(["ts_code", "trade_date_stamp"],axis = 1, inplace = True)
    cols = [i+'_index' for i in index_df.columns.values ]
    index_df.columns = cols
    index_df.rename(columns = {"trade_date_index":"trade_date"}, inplace = True)
    all_df = pd.merge(left = stock_df, right = index_df, how= "left"  ,on= "trade_date")
    all_df.fillna(method= "ffill", inplace=True)
    return all_df

def getStock(stock_code,start_date, end_date  , LOCAL = True):
    daily_df = getStockDaily(stock_code,start_date, end_date  , LOCAL = LOCAL)
    weekly_df = getStockWeekly(stock_code,start_date , end_date  , LOCAL = LOCAL)
    monthly_df = getStockMonthly(stock_code,start_date , end_date  , LOCAL = LOCAL)
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

def getIndex(stock_code,start_date, end_date , LOCAL = True, merge_daily = True):
    if merge_daily:
        daily_df = getIndexDaily(stock_code,start_date, end_date  , LOCAL = True)
        daily_df = getKdj(daily_df)
        daily_df = getMacd(daily_df)
    weekly_df = getIndexWeekly(stock_code,start_date , end_date  , LOCAL = True)
    monthly_df = getIndexMonthly(stock_code,start_date , end_date  , LOCAL = True)
    # KDJ
    
    weekly_df = getKdj(weekly_df)
    weekly_df = getMacd(weekly_df)
    monthly_df = getKdj(monthly_df)
    monthly_df = getMacd(monthly_df)
    # merge
    if merge_daily:
        all_df = mergeDailyWeeklyMonthly(daily_df,weekly_df,monthly_df)
    else:
        all_df = mergeWeeklyMonthly(weekly_df,monthly_df)
    
    return all_df

def getIndustry(stock_code,start_date = "20100101", end_date  = "20200314"  , LOCAL = True, merge_daily = True):
    if merge_daily:
        daily_df = getIndustryDaily(stock_code,start_date, end_date )
        daily_df = getKdj(daily_df)
        daily_df = getMacd(daily_df)
    weekly_df = getIndustryWeekly(stock_code,start_date , end_date )
    monthly_df = getIndustryMonthly(stock_code,start_date , end_date )
    # KDJ and MACD
    
    weekly_df = getKdj(weekly_df)
    weekly_df = getMacd(weekly_df)
    monthly_df = getKdj(monthly_df)
    monthly_df = getMacd(monthly_df)
    # merge
    if merge_daily:
        all_df = mergeDailyWeeklyMonthly(daily_df,weekly_df,monthly_df)
    else:
        all_df = mergeWeeklyMonthly(weekly_df,monthly_df)
    return all_df

