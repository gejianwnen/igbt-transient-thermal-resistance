# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:57:20 2020

@author: gejianwen
"""
# 引入 datetime 模块
import datetime
import time

__all__ = ["getToday","getYesterday",
           "time2str",]


def getToday():
    return time.strftime('%Y%m%d',time.localtime(time.time())) # 当前日期

def getYesterday(): 
    today=datetime.date.today() 
    oneday=datetime.timedelta(days=1) 
    yesterday=today-oneday  
    
    return time2str(yesterday)


def time2str(t,form = "%Y%m%d"):
    # datetime类是date和time的结合体，包括date与time的所有信息，date和time类中具有的方法和属性，datetime类都具有。
    if(type(t) == datetime.datetime):
        return t.strftime("%Y-%m-%d %H:%M:%S")
    if(type(t) == datetime.date):
        return t.strftime(form)
    if(type(t) == time.struct_time): 
        return time.strftime(form,t)
    else:
        try:
            timestr = time.strftime(form,time.localtime(t))
            return timestr
        except Exception as e:
            print(e)
            return 0


