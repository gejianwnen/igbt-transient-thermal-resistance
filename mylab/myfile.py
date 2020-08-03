# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:10:57 2020

@author: admin
"""

import os


def deleteBySize(folder = os.getcwd(),minSize = 1):
    """删除小于minSize的文件（单位：K）"""
    for foldername,subfolders,filenames in os.walk(folder):
        for file in filenames:
            if os.path.getsize(file) < minSize * 1000:
                os.remove(file)    #删除文件
                print(file + " deleted")
    return

 
def deleteNullFile(folder):
    '''删除所有大小为0的文件'''
    for foldername,subfolders,filenames in os.walk(folder):
        for file in filenames:
            if os.path.getsize(file)  == 0:   #获取文件大小
                os.remove(file)
                print(file + " deleted.")
    return

