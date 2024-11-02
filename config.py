import os

main_path = r'E:\Crpto\Data\history_candle_data'
下载线程数 = 48
debug_mode = True  # 调试模式，开启后仅下载前五个交易对，用于调试
interval = '1m'  # 下载K线的周期,请勿修改此参数，因为要计算avg_price_1m，最终得到的K线数据是1H的

proxies = {
    'http': 'http://127.0.0.1:18320',  # 代理设置，根据科学上网工具的端口自行设置
    'https': 'http://127.0.0.1:18320',  # 代理设置，根据科学上网工具的端口自行设置
}

calc_stop = True  # 是否计算止盈止损触发状态列，True为计算，False为不计算
stop_profit_list = [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 100]
stop_loss_list = [-0.02, -0.05, -0.08, -0.1, -0.12, -0.15, -0.2, -0.3, -0.5, -1]

# 以下参数无需修改
现货临时下载文件夹 = os.path.join(main_path, 'Download', 'spot')
永续合约临时下载文件夹 = os.path.join(main_path, 'Download', 'swap')
现货K线存放路径 = os.path.join(main_path, 'spot_binance_1h')
永续合约K线存放路径 = os.path.join(main_path, 'swap_binance_1h')
