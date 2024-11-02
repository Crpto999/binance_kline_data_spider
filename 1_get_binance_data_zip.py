# 1_get_binance_data_zip.py
# -*- coding: utf-8 -*-
import concurrent.futures
import hashlib
import sys
from datetime import datetime, timedelta
from glob import glob
import pandas as pd
import requests
from tqdm import tqdm
from config import *
import random
from pathlib import Path
import threading  # 引入threading模块
from joblib import Parallel, delayed

# 日志锁
log_lock = threading.Lock()


def extract_coin_names(folder_path):
    zip_files = glob(os.path.join(folder_path, '*.zip'))
    coin_names = set()
    for zip_file in zip_files:
        coin_name = os.path.basename(zip_file).split('-')[0]
        coin_names.add(coin_name)
    coin_names_list = list(coin_names)
    coin_names_list.sort()
    return coin_names_list


def get_all_symbols(proxies, target):
    max_retries = 5
    retries = 0
    # 根据 target 的值选择合适的 URL
    url = 'https://data-api.binance.vision/api/v3/exchangeInfo' if target == 'spot' else 'https://fapi.binance.com/fapi/v1/exchangeInfo'

    while retries < max_retries:
        try:
            # 发送请求到 Binance API
            response = requests.get(url, proxies=proxies)
            response.raise_for_status()  # 检查请求是否成功
            data = response.json()  # 解析 JSON 数据
            symbols = [market['symbol'] for market in data['symbols'] if market['symbol'].endswith('USDT')]
            return symbols

        except requests.exceptions.Timeout as e:

            print(f"请求超时，正在重试...({retries + 1}/{max_retries})", e)
            retries += 1  # 重试次数增加

        except requests.exceptions.RequestException as e:

            print(f"网络或请求错误，正在重试...({retries + 1}/{max_retries})", e)
            retries += 1
    return []  # 所有重试尝试后仍然失败，返回空列表


def get_active_symbols(proxies, target):
    try:
        # 发送请求到 Binance API  https://api.binance.com/api/v3/exchangeInfo
        res_spot = requests.get('https://data-api.binance.vision/api/v3/exchangeInfo', proxies=proxies)
        res_swap = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo', proxies=proxies)
        res_spot.raise_for_status()  # 检查请求是否成功
        res_swap.raise_for_status()  # 检查请求是否成功
        data_spot = res_spot.json()  # 解析 JSON 数据
        data_swap = res_swap.json()  # 解析 JSON 数据
        symbols = []

        if target == 'spot':
            # 处理现货市场的交易对
            for market in data_spot['symbols']:
                if market['symbol'].endswith('USDT') and market['status'] == 'TRADING':
                    symbols.append(market['symbol'])

        elif target == 'swap':
            # 处理永续合约市场的交易对
            for market in data_swap['symbols']:
                if market['symbol'].endswith('USDT') and market['status'] == 'TRADING':
                    symbols.append(market['symbol'])

    except requests.exceptions.Timeout as e:
        print("请求超时:", e)
        return []
    except requests.exceptions.RequestException as e:
        print("网络或请求错误:", e)
        return []

    return symbols


def is_full_month(date_range, year, month):
    month_start = datetime(year, month, 1)
    if month == 12:
        month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = datetime(year, month + 1, 1) - timedelta(days=1)

    current_date = month_start
    while current_date <= month_end:
        if current_date not in date_range:
            return False
        current_date += timedelta(days=1)
    return True


def is_url_accessible(btcurl):
    try:
        response = requests.head(btcurl, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False


# 下载URL的函数
def download_url(url, directory, proxies, max_retries=10):
    retries = 0

    while retries < max_retries:
        filename = url.split('/')[-1]
        file_path = os.path.join(directory, filename)

        try:
            response = requests.get(url, proxies=proxies, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return retries
        except requests.RequestException as e:
            if "Not Found for url" in str(e):  # 检查错误信息是否包含 "Not Found for url"
                # print(f"{filename.split('.zip')[0]},此时期无K线数据")
                return -1
            else:
                retries += 1
    return retries


# 主下载逻辑
def download_and_check(url, directory, proxies, max_retries=10):
    result = download_url(url, directory, proxies, max_retries)
    filename = url.split('/')[-1]
    file_path = os.path.join(directory, filename)

    if result == max_retries:
        return url, 'failed', result  # 下载最终失败
    elif os.path.exists(file_path) and result <= max_retries:
        return url, 'success', result  # 下载成功
    else:
        return url, 'retry', result  # 下载后续需要重试


def main_download(urls, directory, proxies):
    # 使用 joblib 并行处理
    results = Parallel(n_jobs=下载线程数)(
        delayed(download_and_check)(url, directory, proxies) for url in urls
    )

    error_urls = []
    retryed_urls = []
    success_urls = []

    for url, status, retries in results:
        filename = url.split('/')[-1]
        if status == 'failed':  # 检查是否为下载失败
            error_urls.append(f'！！！{filename}最终下载失败！！！,重试{retries}次')
        if retries == -1:
            error_urls.append(f'！！！{filename}下载失败！！！,不存在此链接')
        elif status == 'retry':
            retryed_urls.append(f'{filename}下载成功,重试次数：{retries}')
        elif status == 'success':
            success_urls.append(url)

    return error_urls, retryed_urls, success_urls


def log_failed_urls(log_file, urls):
    with log_lock:  # 使用锁确保线程安全
        with open(log_file, 'a', encoding='utf-8') as f:
            for url in urls:
                f.write(f'{url}\n')


def verify_checksum(zip_file_path, checksum_file_path):
    sha256_hash = hashlib.sha256()
    with open(zip_file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    zip_file_hash = sha256_hash.hexdigest()

    with open(checksum_file_path, 'r') as file:
        checksum_file_hash = file.read().split()[0]

    return zip_file_hash == checksum_file_hash


def all_merge_csv(folder_path):
    matching_files = list(Path(folder_path).glob(f"*{'_merge'}*.csv"))
    merge_files = set()  # 使用集合来避免重复的币种名称

    # 统计已存在多少个 _merge 的文件
    existing_merge_files_count = len(matching_files)
    if existing_merge_files_count > 0:
        print(f"检查到上次有任务中断。上次已完成 {existing_merge_files_count} 个币种的清洗合并任务，开始继续下载.")

    for merge_file in matching_files:
        coin_name = os.path.basename(merge_file).split('_')[0]
        merge_files.add(coin_name)

    merge_files_list = list(merge_files)
    merge_files_list.sort()

    # 删除除了带有 _merge 后缀的文件之外的其他 CSV 文件
    for file in Path(folder_path).glob("*.csv"):
        if "_merge" not in file.name:
            file.unlink()

    return merge_files_list


if __name__ == '__main__':
    # 默认值
    target = 'swap'
    # 检查是否有足够的命令行参数
    if len(sys.argv) > 1:
        target = sys.argv[1]

    # 你可以在这里根据target的值进行相应的操作
    if target == "spot":
        download_directory = 现货临时下载文件夹
        data_directory = 现货K线存放路径
        base_url = 'https://data.binance.vision/data/' + target
        mode = "现货"
    else:
        download_directory = 永续合约临时下载文件夹
        data_directory = 永续合约K线存放路径
        base_url = 'https://data.binance.vision/data/futures/um'
        mode = "合约"
    checksum_directory = os.path.join(download_directory, 'checksums')
    os.makedirs(checksum_directory, exist_ok=True)
    # 设置增量zip文件下载目录
    failed_symbols_log = os.path.join(main_path, f'{mode}_Download_failed_log.txt')
    retryed_symbols_log = os.path.join(main_path, f'{mode}_Download_retryed_log.txt')
    Verify_times_log = os.path.join(main_path, f'{mode}_Verify_checksum_times_log.txt')
    # 为每个币种生成URL
    print(f"即将下载 {mode}K线数据")
    print(f'使用的下载接口为:{base_url}')
    symbols = get_all_symbols(proxies, target)  # 下载全部币种,包括现在已经下架的
    # 读取CSV文件，获取最新的candle_begin_time日期
    csv_path = os.path.join(data_directory, 'ZEN-USDT.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, skiprows=1, encoding='GBK')
        latest_date_str = df['candle_begin_time'].max()
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d %H:%M:%S')
        print(f'币安在交易的{target}USDT交易对币种总数:', len(symbols))
    else:
        latest_date = datetime(2017, 9, 3)
        # 生成币种列表

    if len(symbols) < 1:
        exit()

    print(f'币安全部{mode}USDT交易对币种个数:', len(symbols))
    symbols = [symbol for symbol in symbols if not any(keyword in symbol for keyword in ['UP', 'DOWN', 'BEAR', 'BULL'])]
    print(f'去除杠杆代币后的{mode}币种个数:', len(symbols))
    symbols.sort()
    if debug_mode:
        symbols = symbols[:5]  # 调试语句

    # ===指定下载列表的中断点，用于意外中断后的续传
    coins_already_download = extract_coin_names(download_directory)

    if len(coins_already_download) > 0:
        index_next = symbols.index(coins_already_download[-1])
        symbols = symbols[index_next:]

    print(f'即将下载的{mode}币种总个数:', len(symbols))

    merges = all_merge_csv(download_directory)
    if merges:
        symbols = sorted(list(set(symbols) - set(merges)))
        print('需要补充下载的币种:', symbols)
        print('需要补充下载的币种个数:', len(symbols))

    # 获取当前日期
    current_date = datetime.now()

    # 计算日期范围
    start_date = latest_date - timedelta(days=2)

    date_range = [start_date + timedelta(days=i) for i in range((current_date - start_date).days + 1)]
    print(f'下载{mode}K线数据的日期起点:', start_date)
    print(f'下载{mode}K线数据的日期终点:', date_range[-1])

    # 计算上一个月的年份和月份
    current_year, current_month = datetime.now().year, datetime.now().month
    previous_year, previous_month = (current_year - 1, 12) if current_month == 1 else (current_year, current_month - 1)

    # 构建上一个月的月度数据URL
    previous_month_url = f"{base_url}/monthly/klines/BTCUSDT/{interval}/BTCUSDT-{interval}-{previous_year}-{str(previous_month).zfill(2)}.zip"

    # 检查上一个月的URL是否可访问
    previous_month_accessible = is_url_accessible(previous_month_url)

    # 分组为月度下载和日度下载
    monthly_download = set()

    daily_download = []
    for date in date_range:
        year_month = (date.year, date.month)
        if is_full_month(date_range, date.year, date.month) and (
                year_month != (previous_year, previous_month) or previous_month_accessible):
            monthly_download.add(year_month)
        else:
            daily_download.append(date)
    monthly_download = sorted(monthly_download, key=lambda x: (x[0], x[1]))
    daily_download.sort()

    pbar = tqdm(symbols, desc=f"📈 开始下载{symbols[0]}...", unit=f"{mode}")
    for symbol in pbar:
        urls = []
        checksum_urls = []
        # 添加月度数据URL
        for year, month in monthly_download:
            url = f"{base_url}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{str(month).zfill(2)}.zip"
            urls.append(url)
            checksum_urls.append(url + '.CHECKSUM')
        # # 添加日度数据URL
        for date in daily_download:
            date_str = date.strftime('%Y-%m-%d')
            url = f"{base_url}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
            urls.append(url)
            checksum_urls.append(url + '.CHECKSUM')

        # 去重URLs
        urls = list(set(urls))
        urls.sort()

        checksum_urls = list(set(checksum_urls))
        checksum_urls.sort()

        # 下载失败的url列表
        failed_symbols, retryed_symbols, success_symbol_urls = main_download(urls, download_directory, proxies)
        # 下载失败的CHECKSUM文件的url列表
        failed_checksums, retryed_checksums, success_checksums = main_download(checksum_urls, checksum_directory, proxies)

        # 初始化用于跟踪每个币种失败次数的字典
        if len(failed_symbols) > 0:
            log_failed_urls(failed_symbols_log, failed_symbols)

        if len(retryed_symbols) > 0:
            log_failed_urls(retryed_symbols_log, retryed_symbols)

        for url in success_symbol_urls:
            filename = url.split('/')[-1]
            zip_file_path = os.path.join(download_directory, filename)
            checksum_file_path = os.path.join(checksum_directory, filename + '.CHECKSUM')

            valid = False
            attempts = 0
            max_attempts = 10
            while not valid and attempts < max_attempts:
                if os.path.exists(zip_file_path) and os.path.exists(checksum_file_path):
                    # 使用verify_checksum函数验证文件
                    valid = verify_checksum(zip_file_path, checksum_file_path)

                    if not valid:
                        # 如果校验失败，记录该币种的失败次数
                        symbol = filename.split('_')[0]

                        # 删除原有文件，以便重新下载
                        os.remove(zip_file_path)
                        os.remove(checksum_file_path)

                        # 重新下载ZIP文件和CHECKSUM文件
                        download_url(url, download_directory, proxies)
                        download_url(url + '.CHECKSUM', checksum_directory, proxies)

                attempts += 1
            # 校验结束，记录重试次数大于1的情况
            if attempts > 1:  # 等于1时，代表经过一次校验即通过
                with open(Verify_times_log, 'a', encoding='utf-8') as f:  # 使用追加模式'a'
                    f.write(f"{symbol}: {filename}, 校验重试次数: {attempts - 1}\n")

        matching_files = list(Path(download_directory).glob(f"*{symbol}*.zip"))
        num_matching_files = len(matching_files)
        emoji_options = ["✅", "🎉", "🌟", "🚀", "💡", "🔥", "🌈", "💎", "😎", "🌸", ]
        random_emoji = random.choice(emoji_options)
        coin_name = symbol.replace("USDT", "-USDT")
        pbar.set_description(f"{random_emoji}{coin_name} 成功下载并通过校验,包含{num_matching_files}个.zip文件")

    pbar.close()
