# -*- coding: utf-8 -*-
import sys
import time
import zipfile
from glob import glob
from itertools import product
from pathlib import Path
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from config import *

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行


# 通过所有zip文件的文件名前缀获取币种名称列表
def extract_coin_names(folder_path):
    zip_files = glob(os.path.join(folder_path, '*.zip'))
    print(f'发现 {len(zip_files)} 个{mode}zip 文件.')
    time.sleep(1)
    coin_names = set()  # 使用集合来避免重复的币种名称
    for zip_file in zip_files:
        coin_name = os.path.basename(zip_file).split('-')[0]
        coin_names.add(coin_name)
    coin_names_list = list(coin_names)
    coin_names_list.sort()
    return coin_names_list


def all_merge_csv(folder_path):
    matching_files = list(Path(folder_path).glob(f"*{'_merge'}*.csv"))
    merge_files = set()  # 使用集合来避免重复的币种名称

    # 统计已存在多少个 _merge 的文件
    existing_merge_files_count = len(matching_files)
    if existing_merge_files_count > 2:
        print(f"检查到上次有任务中断。上次已完成 {existing_merge_files_count} 个币种的清洗任务，开始续洗.")

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
    else:
        return []


# 解压并删除zip文件
def unzip_and_delete_zip(zip_files, folder_path):
    # 查找指定币种的zip文件
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder_path)  # 解压
        # 删除原zip文件
        # os.remove(zip_file)
    # print(f"{coin_name}解压完成")


def process_single_file(file):
    # 读取文件的第一行
    with open(file, 'r') as f:
        first_line = f.readline().strip()  # .strip() 移除可能的前后空白字符

    # 检查第一行是否包含任何预期的列名
    has_header = any(col_name in first_line for col_name in ["open_time", "open", "high", "low", "close"])

    # 根据文件是否有列名来读取数据
    df = pd.read_csv(file, header=0 if has_header else None)

    # 币安API返回的部分文件没有列名，如果没有列名，需要手动指定
    if not has_header:
        df.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ]
    # 将列名映射到新的列名
    column_mapping = {
        "open_time": "candle_begin_time",
        "count": "trade_num",
        "taker_buy_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume"
    }
    df.rename(columns=column_mapping, inplace=True)

    # 添加新的列
    df['symbol'] = file.split(os.sep)[-1].split('USDT')[0] + '-USDT'
    # 注意：avg_price_1m 和 avg_price_5m 需要后续计算

    # 转换时间格式
    df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms')

    # 删除不需要的列
    df.drop(['close_time', 'ignore'], axis=1, inplace=True)

    return df


def process_coin_files(files):
    dataframes = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
        delayed(process_single_file)(file) for file in files
    )
    merged_df = pd.concat(dataframes)

    merged_df.sort_values(by='candle_begin_time', inplace=True)
    merged_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')
    merged_df.reset_index(drop=True, inplace=True)
    # 填充空缺的数据
    start_date = merged_df.iloc[0]['candle_begin_time']
    end_date = merged_df.iloc[-1]['candle_begin_time']
    benchmark = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='1min'))  # 修改频率别名
    benchmark.rename(columns={0: 'candle_begin_time'}, inplace=True)
    merged_df = pd.merge(left=benchmark, right=merged_df, on='candle_begin_time', how='left', sort=True, indicator=True)
    merged_df['close'] = merged_df['close'].ffill()  # 使用推荐的方法
    merged_df['symbol'] = merged_df['symbol'].ffill()  # 使用推荐的方法
    for column in ['open', 'high', 'low']:
        merged_df[column] = merged_df[column].fillna(merged_df['close'])
    _ = ['volume', 'quote_volume', 'trade_num', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    merged_df[_] = merged_df[_].fillna(0)
    # merged_df = merged_df[merged_df['_merge'] == 'left_only']

    # 将索引转换为DatetimeIndex，如果它还不是
    merged_df.set_index('candle_begin_time', inplace=True)

    merged_df['avg_price_1m'] = merged_df['quote_volume'] / merged_df['volume']
    merged_df['avg_price_5m'] = merged_df['quote_volume'].rolling(window=5).sum() / merged_df['volume'].rolling(
        window=5).sum()
    merged_df['avg_price_5m'] = merged_df['avg_price_5m'].shift(-4)
    merged_df['avg_price_1m'] = merged_df['avg_price_1m'].fillna(merged_df['open'])  # 直接赋值
    merged_df['avg_price_5m'] = merged_df['avg_price_5m'].fillna(merged_df['open'])  # 直接赋值

    hourly_df = merged_df.resample('1h').agg({  # 修改频率别名
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'symbol': 'first',
        'avg_price_1m': 'first',
        'avg_price_5m': 'first',
    })
    hourly_df.reset_index(inplace=True)

    return hourly_df


def process_single_combination(df, stop_profit, stop_loss):
    from numpy.lib.stride_tricks import sliding_window_view
    """
    计算特定止盈和止损策略下的触发情况。

    :param df: pandas DataFrame，包含必要的列 ['candle_begin_time', 'high', 'low', 'avg_price_1m', 'open']
    :param stop_profit: float，止盈百分比（例如 0.05 表示 5%）
    :param stop_loss: float，止损百分比（例如 -0.03 表示 3%）
    :return: pandas Series，表示每个时间点的止盈止损触发情况
             0 - 未触发
            -1 - 止损先触发
             1 - 止盈先触发
             2 - 同时触发
    """
    # 确保 DataFrame 按时间排序，并重置索引
    df = df.sort_values('candle_begin_time').reset_index(drop=True)

    # 提取必要的列为 NumPy 数组，便于高效计算
    high = df['high'].values
    low = df['low'].values
    avg_price_1m = df['avg_price_1m'].values
    n = len(df)  # 数据总行数

    # 计算每个时间点的止盈价和止损价
    stop_profit_price = avg_price_1m * (1 + stop_profit)
    stop_loss_price = avg_price_1m * (1 + stop_loss)

    # 定义时间窗口大小为24
    window_size = 24

    # 填充 high 和 low 数据，以确保所有窗口都有完整的数据
    # 使用 'edge' 模式，即使用最后一个有效值进行填充
    pad_width = window_size
    high_padded = np.pad(high, (0, pad_width), mode='edge')  # 形状 (n + window_size,)
    low_padded = np.pad(low, (0, pad_width), mode='edge')  # 形状 (n + window_size,)

    # 使用 sliding_window_view 构建滑动窗口
    # 每个窗口包含24个元素，从当前行开始
    high_windows = sliding_window_view(high_padded, window_shape=window_size)[:n]  # 形状 (n, window_size)
    low_windows = sliding_window_view(low_padded, window_shape=window_size)[:n]  # 形状 (n, window_size)

    # 生成 final_price
    # 对 avg_price_1m 进行填充，以获取每个窗口后的第24个价格
    avg_price_padded = np.pad(avg_price_1m, (0, pad_width + 1), mode='edge')  # 形状 (n + window_size +1,)
    final_price = avg_price_padded[window_size: window_size + n]  # 形状 (n,)

    # 将 final_price 添加为每个窗口的最后一列
    # final_price[:, np.newaxis] 将其形状从 (n,) 转换为 (n, 1) 以便于拼接
    high_full = np.concatenate([high_windows, final_price[:, np.newaxis]], axis=1)  # 形状 (n, window_size + 1)
    low_full = np.concatenate([low_windows, final_price[:, np.newaxis]], axis=1)  # 形状 (n, window_size + 1)

    # ------------------ 触发检测逻辑开始 ------------------
    # 包含 final_price 在内的触发检测
    profit_trigger = high_full > stop_profit_price[:, np.newaxis]  # 形状 (n, window_size +1)
    loss_trigger = low_full < stop_loss_price[:, np.newaxis]  # 形状 (n, window_size +1)

    # 对于每一行，找到第一个触发的位置
    profit_any = np.any(profit_trigger, axis=1)
    profit_indices = np.argmax(profit_trigger, axis=1)
    profit_indices = np.where(profit_any, profit_indices, float('inf'))

    loss_any = np.any(loss_trigger, axis=1)
    loss_indices = np.argmax(loss_trigger, axis=1)
    loss_indices = np.where(loss_any, loss_indices, float('inf'))

    # 计算最终的止盈止损结果
    result = np.zeros(n, dtype=int)

    # 止损先触发
    mask_loss_first = (loss_indices < profit_indices)
    result[mask_loss_first] = -1

    # 止盈先触发
    mask_profit_first = (profit_indices < loss_indices)
    result[mask_profit_first] = 1

    # 同时触发（在同一时间点）
    mask_both_same = (profit_indices == loss_indices) & (profit_indices != float('inf'))
    result[mask_both_same] = 2

    # 将结果转换为 pandas Series
    return pd.Series(result, index=df.index, name=f'stop[{stop_profit}_{stop_loss}]')


def process_stop(df, stop_loss_list, stop_profit_list, calc_stop=True, use_parallel=False, n_jobs=-1):
    if not calc_stop:
        return df

    # Generate all combinations of stop profit and stop loss
    stop_all_combinations = list(product(stop_profit_list, stop_loss_list))

    if use_parallel:
        # Parallel processing
        results_dfs = Parallel(n_jobs=n_jobs)(
            delayed(process_single_combination)(df, stop_profit, stop_loss)
            for stop_profit, stop_loss in stop_all_combinations
        )
    else:
        # Serial processing
        results_dfs = []
        for stop_profit, stop_loss in stop_all_combinations:
            result_df = process_single_combination(df, stop_profit, stop_loss)
            results_dfs.append(result_df)

    # Concatenate all the result DataFrames along the columns
    results_combined = pd.concat(results_dfs, axis=1)

    # Merge the original DataFrame with the results
    df_final = pd.concat([df, results_combined], axis=1)

    return df_final


def get_merge_csv_files(folder_path):
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    grouped_files = {}
    for file in csv_files:

        # if '_merge' in file:
        if '_merged' in os.path.basename(file):
            continue
        coin_name = os.path.basename(file).split('-')[0]

        grouped_files.setdefault(coin_name, []).append(file)

    for coin_name, files in grouped_files.items():
        try:
            hourly_df = process_coin_files(files)

            df_final = process_stop(hourly_df, stop_loss_list, stop_profit_list)

            df_final.to_csv(os.path.join(folder_path, f'{coin_name}_merged.csv'), index=False)

        except Exception as exc:
            print(f"\n {coin_name}生成过程中出错: {exc}")


# 删除未合并的CSV文件
def delete_unmerged_csv_files(folder_path):
    # 查找文件夹中所有的CSV文件
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    for csv_file in csv_files:
        # 检查文件名是否不包含 '_merged'
        if '_merged' not in os.path.basename(csv_file):
            os.remove(csv_file)  # 删除文件


if __name__ == "__main__":
    # 默认值
    target = 'swap'
    # 检查是否有足够的命令行参数
    if len(sys.argv) > 1:
        target = sys.argv[1]
    if target == "spot":
        download_directory = 现货临时下载文件夹
        mode = "现货"
    else:
        download_directory = 永续合约临时下载文件夹
        mode = "合约"

    coins_to_clean = extract_coin_names(download_directory)
    merge_csvs = all_merge_csv(download_directory)

    # 如果任务中断，识别断点，继续清理
    if len(merge_csvs) >= 2:
        index_next_clean = coins_to_clean.index(merge_csvs[-2])
        coins_to_clean = coins_to_clean[index_next_clean:]

    with tqdm(total=len(coins_to_clean), desc="总体进度", unit=mode) as pbar:
        for coin_name in coins_to_clean:
            coin = coin_name.replace("USDT", "-USDT")
            # 步骤1: 解压
            zip_files = glob(os.path.join(download_directory, f'{coin_name}*.zip'))
            file_num = len(zip_files)
            pbar.set_description(f"📦 正在解压{file_num}个{coin}的zip文件")
            unzip_and_delete_zip(zip_files, download_directory)  # 解压指定币种的zip文件并删除

            # 步骤2: 清洗合并
            pbar.set_description(f"🚿 正在清洗合并{file_num}个{coin}的csv文件")
            get_merge_csv_files(download_directory)

            # 步骤3: 删除这个币种的一分钟CSV,完成处理
            delete_unmerged_csv_files(download_directory)
            pbar.update(1)
            pbar.set_description(f"💛 {file_num}个{coin} 的{mode}csv️清洗完成，已合并保存")
            time.sleep(1)
    pbar.close()
