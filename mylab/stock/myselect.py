import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import mpl_finance as mpf
import matplotlib.pyplot as plt

#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = 'SimHei'  # Chinese 


__all__ = ["selectStock1","selectStock2",
           "selectByKdj1",
           ]


def selectByKdj1(df, index_monthly_KDJ = True, index_weekly_KDJ = True):
    """
    select only by weekly and monthly Kdj
    """
    select_df = df.copy(deep = True)
    select_df = select_df.loc[select_df["D_monthly"].values < 60, :]
    select_df = select_df.loc[select_df["D_weekly"].values < 60, :]
    if index_weekly_KDJ:
        select_df = select_df.loc[select_df["K_weekly"].values >= (select_df['D_weekly'].values-1), :]
    else:
        select_df = select_df.loc[select_df["K_weekly"].values < select_df['D_weekly'].values, :]
    if index_monthly_KDJ:
        select_df = select_df.loc[select_df["K_monthly"].values >= (select_df['D_monthly'].values-1), :]
    else:
        select_df = select_df.loc[select_df["K_monthly"].values < select_df['D_monthly'].values, :]
    
    return select_df

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

