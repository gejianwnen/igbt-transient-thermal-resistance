# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:15:09 2020

@author: gejianwen
"""

import win32gui
import win32con
import win32clipboard as w
import time

__all__ = ["sendQQ"]

def sendQQ(name, msg):
    #打开剪贴板
    w.OpenClipboard()
    #清空剪贴板
    w.EmptyClipboard()
    #设置剪贴板内容
    w.SetClipboardData(win32con.CF_UNICODETEXT, msg)
    #获取剪贴板内容
    date = w.GetClipboardData()
    #关闭剪贴板
    w.CloseClipboard()
    #获取qq窗口句柄
    handle = win32gui.FindWindow(None, name)
    if handle == 0:
        print('未找到窗口！')
    #显示窗口
    win32gui.ShowWindow(handle,win32con.SW_SHOW)
    #把剪切板内容粘贴到qq窗口
    win32gui.SendMessage(handle, win32con.WM_PASTE, 0, 0)
    #按下后松开回车键，发送消息
    win32gui.SendMessage(handle, win32con.WM_KEYDOWN, win32con.VK_RETURN, 0)
    win32gui.SendMessage(handle, win32con.WM_KEYUP, win32con.VK_RETURN, 0)
    time.sleep(1)#延缓进程