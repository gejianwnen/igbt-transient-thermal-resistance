# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:19:13 2020

@author: gejianwen
"""

from email.mime.text import MIMEText
import smtplib

def loginEmial(smtp_addr = "smtp.139.com", sender = "13061799343@139.com", password = "951019Ge"):
    try:
        smtp_server = smtplib.SMTP_SSL(smtp_addr, 465) # 第一个参数：smtp服务地址（你发送邮件所使用的邮箱的smtp地址，在网上可以查到，比如qq邮箱为smtp.qq.com） 第二个参数：对应smtp服务地址的端口号
        smtp_server.login(sender, password) # 第一个参数：发送者的邮箱账号 第二个参数：对应邮箱账号的密码
    except Exception as e:
        print(e)
        return None
    return smtp_server


def sendEmail(subject = "买入股票",mail_content = "", sender = "13061799343@139.com", reveivers = "18217565517@139.com" ):
    try:
        content = MIMEText(mail_content, 'plain', 'utf-8') 
        # 第一个参数：邮件的内容；第二个参数：邮件内容的格式，普通的文本，可以使用:plain,如果想使内容美观，可以使用:html；
        #第三个参数：设置内容的编码，这里设置为:utf-8
        content['To'] = reveivers 
        # 设置邮件的接收者，多个接收者之间用逗号隔开
        content['From'] = sender # 邮件的发送者,最好写成str("这里填发送者")，不然可能会出现乱码
        content['Subject'] = subject # 邮件的主题
        smtp_server = smtplib.SMTP_SSL("smtp.139.com", 465) # 第一个参数：smtp服务地址（你发送邮件所使用的邮箱的smtp地址，在网上可以查到，比如qq邮箱为smtp.qq.com） 第二个参数：对应smtp服务地址的端口号
        smtp_server.login(sender, "951019Ge") # 第一个参数：发送者的邮箱账号 第二个参数：对应邮箱账号的密码
        smtp_server.sendmail(sender, [reveivers], content.as_string()) # 第一个参数：发送者的邮箱账号；第二个参数是个列表类型，每个元素为一个接收者；第三个参数：邮件内容
        smtp_server.quit() # 发送完成后加上这个函数调用，类似于open文件后要跟一个close文件一样
    except Exception as e:
        print(e)
        return 0
    return 1