# -*- coding: utf-8 -*-
import subprocess
from config import *
import shutil

'''
接口来源：币安
官方接口github：https://github.com/binance/binance-public-data
数据来源：https://data.binance.vision
'''
要下载的数据类型 = ['swap', 'spot']  # 'swap',#'spot'

scripts = [
    ('1_get_binance_data_zip.py', True),  # 需要模式参数
    ('2_release_zip_and_clean_data.py', True),  # 不需要模式参数
    ('3_merge_to_orginal_csv.py', True)  # 需要模式参数
]

for mode in 要下载的数据类型:
    if mode == 'spot':
        os.makedirs(现货K线存放路径, exist_ok=True)
        os.makedirs(现货临时下载文件夹, exist_ok=True)
    elif mode == 'swap':
        os.makedirs(永续合约K线存放路径, exist_ok=True)
        os.makedirs(永续合约临时下载文件夹, exist_ok=True)
    for script in scripts:
        subprocess.run(['python', script[0], mode])

# 可以在脚本运行结束后删除临时下载文件夹
temp_folder = os.path.dirname(现货临时下载文件夹)
shutil.rmtree(temp_folder)
print(f"下载临时文件夹 {temp_folder}已被删除")
