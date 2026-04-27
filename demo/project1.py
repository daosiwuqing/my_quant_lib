# 该项目的作用: 实现期货策略tick级别的回测
import pandas as pd
import numpy as np
import queue, threading, bisect

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.animation import FuncAnimation

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from package.MyTT import *
import my_quant_lib as MQL

# 设置全局字体
fm.fontManager.addfont(
    "C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)
mpl.rcParams["font.sans-serif"] = "FZHei-B01S"
mpl.rcParams["axes.unicode_minus"] = False
font = fm.FontProperties(
    fname="C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)


def b_ma_tangled(row, ma_columns, threshold=0.01):
    """
    功能: 判断当前周期均线是否纠缠

    参数:
        row(pd.Series): DataFrame的一行
        ma_columns(list): 均线列名列表
        threshold(float): 百分比阈值, 例如0.01表示1%

    返回:
        True表示纠缠, False表示不纠缠
    """
    ma_values = row[ma_columns].dropna().values
    if len(ma_values) < 2:
        return False
    max_ma = np.max(ma_values)
    min_ma = np.min(ma_values)

    return (max_ma - min_ma) / min_ma <= threshold


def volume_weighted_price(open, high, low, close, vol, window=14):
    """
    功能: 计算成交量加权价格

    参数:
        open(pd.Series): 开盘价
        high(pd.Series): 最高价
        low(pd.Series): 最低价
        close(pd.Series): 收盘价
        vol(pd.Series): 成交量
        window(int): 滚动窗口大小, 默认为14

    返回:
        result(pd.Series): 成交量加权价格
    """
    typical_price = (open + high + low + close) / 4  # 计算典型价格 (每根K线的平均价)
    weighted_volume = typical_price * vol  # 计算加权和: 典型价格 * 成交量

    # 滚动求和
    sum_weighted = weighted_volume.rolling(window=window).sum()
    sum_volume = vol.rolling(window=window).sum()

    denominator = sum_volume.replace(0, float("nan"))  # 防止除以零
    result = sum_weighted / denominator  # 计算结果

    return result


path1 = "D:\\LearningAndWorking\\VSCode\\python\\project2\\RB_main_contract.csv"
df1_1 = pd.read_csv(path1)
df1_1["date"] = pd.to_datetime(df1_1["date"])
main_contract_dict = {}
for contract in sorted(set(df1_1["contract"])):
    df1_2 = df1_1[df1_1["contract"] == contract]
    start_date = df1_2["date"].min()
    end_date = df1_2["date"].max()
    main_contract_dict[contract] = [start_date, end_date]


path2 = "D:\\LearningAndWorking\\VSCode\\python\\project2\\RB_main_contract\\rb_2020_main_contract.parquet"
df2_1 = pd.read_parquet(path2)
selected_keys = df2_1["instrument_id"].unique()
filtered_dict = {
    k2: main_contract_dict[k2]
    for k2 in main_contract_dict
    if k2.lower() in [k1.lower() for k1 in selected_keys]
}
for key, value in filtered_dict.items():
    df2_2 = df2_1[df2_1["instrument_id"].str.lower() == str(key).lower()]
    time_ranges = [
        ("08:30:00", "09:00:00"),
        ("10:15:00", "10:30:00"),
        ("11:30:00", "12:00:00"),
        ("15:00:00", "15:30:00"),
        ("20:30:00", "21:00:00"),
        ("23:00:00", "23:30:00"),
    ]
    for start, end in time_ranges:
        start_time = pd.to_timedelta(start)
        end_time = pd.to_timedelta(end)
        df2_2 = df2_2[
            ~((df2_2["update_time"] >= start_time) & (df2_2["update_time"] < end_time))
        ]
    df2_2 = df2_2[
        [
            "calender_day",
            "update_time",
            "update_millisec",
            "instrument_id",
            "last_price",
            "volume",
            "open_interest",
        ]
    ]
    df2_2.insert(
        loc=0, column="datetime", value=df2_2["calender_day"] + df2_2["update_time"]
    )
    df2_2 = df2_2.sort_values(by=["datetime"])

    # 1, 向量化回测
    # 处理k线数据
    df3_1 = MQL.StaticTickToK.tick_to_K(df2_2, "15min")
    df3_1 = df3_1.reset_index()

    df3_1["ma5"] = MA(df3_1["close"], 5)
    df3_1["ma10"] = MA(df3_1["close"], 10)
    df3_1["ma20"] = MA(df3_1["close"], 20)
    df3_1["ma40"] = MA(df3_1["close"], 40)
    df3_1["ma60"] = MA(df3_1["close"], 60)

    df3_1["local_min"] = (
        df3_1["low"]
        .rolling(window=13)
        .apply(MQL.OtherTools.find_local_extrema, raw=False, args=(13, 9, "low"))
    )
    df3_1.loc[df3_1["local_min"] == False, "local_min"] = np.nan
    df3_1.loc[df3_1["local_min"] == True, "local_min"] = df3_1.loc[
        df3_1["local_min"] == True, "low"
    ]
    df3_1["local_max"] = (
        df3_1["high"]
        .rolling(window=13)
        .apply(MQL.OtherTools.find_local_extrema, raw=False, args=(13, 9, "high"))
    )
    df3_1.loc[df3_1["local_max"] == False, "local_max"] = np.nan
    df3_1.loc[df3_1["local_max"] == True, "local_max"] = df3_1.loc[
        df3_1["local_max"] == True, "high"
    ]

    df3_1["close_open"] = df3_1["close"] / df3_1["open"] - 1
    df3_1["close_open_0.05"] = (
        df3_1["close_open"]
        .expanding()
        .apply(MQL.OtherTools.find_per_value, raw=False, args=(0.05,))
    )
    df3_1["close_open_0.95"] = (
        df3_1["close_open"]
        .expanding()
        .apply(MQL.OtherTools.find_per_value, raw=False, args=(0.95,))
    )
    df3_1["volume_0.95_percentile"] = (
        df3_1["volume"]
        .expanding()
        .apply(MQL.OtherTools.find_per_value, raw=False, args=(0.95,))
    )
    df3_1["long_stop_profit"] = (df3_1["close_open"] >= df3_1["close_open_0.95"]) & (
        df3_1["volume"] >= df3_1["volume_0.95_percentile"]
    )
    df3_1["short_stop_profit"] = (df3_1["close_open"] <= df3_1["close_open_0.05"]) & (
        df3_1["volume"] >= df3_1["volume_0.95_percentile"]
    )

    df3_1["long_stop_price"] = df3_1["low"].shift(1)
    df3_1["short_stop_price"] = df3_1["high"].shift(1)

    spread_list = []
    pre_local_min = 0
    pre_local_max = 0
    spread = np.nan
    for row in df3_1.itertuples(index=False):
        local_min = row.local_min
        local_max = row.local_max

        if ~np.isnan(local_min) and pre_local_min != local_min:
            if pre_local_max == 0:
                spread_list.append(spread)
                pre_local_min = local_min
            else:
                spread = pre_local_max - local_min
                spread_list.append(spread)
                pre_local_min = local_min
        elif ~np.isnan(local_max) and pre_local_max != local_max:
            if pre_local_min == 0:
                spread_list.append(spread)
                pre_local_max = local_max
            else:
                spread = local_max - pre_local_min
                spread_list.append(spread)
                pre_local_max = local_max
        else:
            spread_list.append(spread)
    df3_1["spread"] = spread_list

    df3_1["atr"] = ATR(df3_1["close"], df3_1["high"], df3_1["low"], 13)

    df3_1["rsi"] = RSI(df3_1["close"], 13)
    df3_1["rsi_0.05_percentile"] = (
        df3_1["rsi"]
        .expanding()
        .apply(MQL.OtherTools.find_per_value, raw=False, args=(0.05,))
    )
    df3_1["rsi_0.95_percentile"] = (
        df3_1["rsi"]
        .expanding()
        .apply(MQL.OtherTools.find_per_value, raw=False, args=(0.95,))
    )
    df3_1["long_rsi"] = df3_1["rsi"] <= df3_1["rsi_0.05_percentile"]
    df3_1["short_rsi"] = df3_1["rsi"] >= df3_1["rsi_0.95_percentile"]

    ma_columns = ["close", "ma5", "ma10", "ma20"]
    df3_1["ma_tangled"] = df3_1.apply(
        b_ma_tangled, axis=1, ma_columns=ma_columns, threshold=0.005
    )  # 判断每根K线是否均线纠缠
    df3_1["tangled_count"] = (
        df3_1["ma_tangled"].rolling(window=13).sum()
    )  # 检查最近window根K线是否连续纠缠
    df3_1["b_continuously_tangled"] = (
        df3_1["tangled_count"] >= 0.75 * 13
    )  # 如果最近window根K线都为True (即sum==window) , 则判定为 "均线持续纠缠"

    df3_1["vwap20"] = volume_weighted_price(
        df3_1["open"],
        df3_1["high"],
        df3_1["low"],
        df3_1["close"],
        df3_1["volume"],
        window=20,
    )

    df3_2 = MQL.StaticTickToK.tick_to_K(df2_2, "1D")
    df3_2 = df3_2.reset_index()
    df3_2 = df3_2[["trading_day", "low", "high"]]
    df3_2 = df3_2.rename(columns={"low": "yd_low", "high": "yd_high"})

    df3_3 = pd.merge(
        df3_1, df3_2, left_on=["datetime"], right_on=["trading_day"], how="outer"
    )
    df3_1 = df3_3[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "ma5",
            "ma10",
            "ma20",
            "ma40",
            "ma60",
            "local_min",
            "local_max",
            "spread",
            "long_stop_profit",
            "short_stop_profit",
            "long_stop_price",
            "short_stop_price",
            "atr",
            "long_rsi",
            "short_rsi",
            "b_continuously_tangled",
            "vwap20",
            "yd_low",
            "yd_high",
        ]
    ]

    # 观察交易信号
    # spread_list = []
    # pre_local_min = 0
    # pre_local_max = 0
    # spread = np.nan
    # for row in df3_1.itertuples(index=False):
    #     local_min = row.local_min
    #     local_max = row.local_max

    #     if ~np.isnan(local_min) and pre_local_min!=local_min:
    #         if pre_local_max == 0:
    #             spread_list.append(spread)
    #             pre_local_min = local_min
    #         else:
    #             spread = pre_local_max - local_min
    #             spread_list.append(spread)
    #             pre_local_min = local_min
    #     elif ~np.isnan(local_max) and pre_local_max!=local_max:
    #         if pre_local_min == 0:
    #             spread_list.append(spread)
    #             pre_local_max = local_max
    #         else:
    #             spread = local_max - pre_local_min
    #             spread_list.append(spread)
    #             pre_local_max = local_max
    #     else:
    #         spread_list.append(spread)
    # df3_1.loc[df3_1.index, "spread"] = spread_list
    # df3_2 = df3_1[(df3_1['datetime'].between(value[0], value[1]))]
    # df3_2.set_index('datetime', inplace=True)
    # if len(df3_2) > 0:
    #     custom_style = mpf.make_mpf_style(
    #             base_mpf_style='classic', # 基础样式
    #             marketcolors=mpf.make_marketcolors(up='red', down='green', # 上涨和下跌的颜色
    #                                                 wick={'up': 'red', 'down': 'green'}, # 烛芯的颜色
    #                                                 edge={'up': 'red', 'down': 'green'}, # 边缘的颜色
    #                                                 volume={'up': 'red', 'down': 'green'}), # 成交量的颜色
    #             gridcolor='gray', # 网格线的颜色
    #             gridstyle='--', # 网格线的样式
    #             facecolor='white', # 背景颜色
    #             figcolor='white', # 图表颜色
    #             edgecolor='white', # 边缘颜色
    #             rc={'font.family': font.get_name(), 'axes.unicode_minus': False}) # 字体格式
    #     fig = mpf.figure(figsize=(13, 10))
    #     gs = fig.add_gridspec(5, 1)
    #     ax1 = fig.add_subplot(gs[0:3, 0])
    #     ax2 = fig.add_subplot(gs[3:4, 0], sharex=ax1)
    #     ax3 = fig.add_subplot(gs[4:5, 0], sharex=ax1)
    #     def animate(i):
    #         ax1.clear()
    #         ax2.clear()
    #         ax3.clear()

    #         start = max(0, min(len(df3_2), i + 1) - 100)
    #         data = df3_2.iloc[start : i + 1]
    #         local_min_list = list(data['local_min'])[4:] + [np.nan]*4
    #         local_max_list = list(data['local_max'])[4:] + [np.nan]*4
    #         spread_list = list(data['spread'])
    #         ma5_list = list(data['ma5'])
    #         ma10_list = list(data['ma10'])
    #         ma20_list = list(data['ma20'])
    #         ma40_list = list(data['ma40'])
    #         ma60_list = list(data['ma60'])
    #         volume_list = list(data['volume'])
    #         open_interest_list = list(data['open_interest'])

    #         mpf.plot(data, type='candle', style=custom_style, ax=ax1, volume=False)
    #         ax1.plot(local_min_list, 'o', markersize=3, color='#000000', label='local_min')
    #         ax1.plot(local_max_list, 'o', markersize=3, color='#000000', label='local_max')
    #         ax1.text(0.5, 0.9, 'spread: ' + str(spread_list[-1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='black')
    #         ax1.plot(ma5_list, color='red', linewidth=1)
    #         ax1.plot(ma10_list, color='yellow', linewidth=1)
    #         ax1.plot(ma20_list, color='green', linewidth=1)
    #         ax1.plot(ma40_list, color='blue', linewidth=1)
    #         ax1.plot(ma60_list, color='purple', linewidth=1)
    #         ax2.bar(range(len(volume_list)), volume_list, color='#DEB887', label='Volume')
    #         ax3.bar(range(len(open_interest_list)), open_interest_list, color='#DEB887', label='OI')

    #         plt.setp(ax1.get_xticklabels(), visible=False)
    #         plt.setp(ax2.get_xticklabels(), visible=False)
    #         plt.tight_layout()

    #     ani = FuncAnimation(fig, animate, frames=range(len(df3_2)), interval=50, repeat=False)
    #     mpf.show()
    #     plt.close('all')

    # 1.1, 策略一: 根据多均线发散来确定方向, 高低点突破进场, 1个ATR止损——>15min可行
    # 1.1.1, 处理tick数据
    # df4_1 = pd.merge(df2_2, df3_1, left_on=['datetime'], right_on=['datetime'], how='outer')
    # df4_1['ma5'] = df4_1['ma5'].ffill()
    # df4_1['ma10'] = df4_1['ma10'].ffill()
    # df4_1['ma20'] = df4_1['ma20'].ffill()
    # df4_1['ma40'] = df4_1['ma40'].ffill()
    # df4_1['ma60'] = df4_1['ma60'].ffill()
    # df4_1['local_min'] = df4_1['local_min'].ffill()
    # df4_1['local_max'] = df4_1['local_max'].ffill()
    # df4_1['atr'] = df4_1['atr'].ffill()
    # df4_1['start_time'] = ~(((df4_1['update_time']>=pd.to_timedelta('20:55:00')) & (df4_1['update_time']<pd.to_timedelta('21:15:00')))
    #                         | ((df4_1['update_time']>=pd.to_timedelta('14:45:00')) & (df4_1['update_time']<=pd.to_timedelta('15:00:00')))) # 限制开仓时间段

    # 1.1.2, 筛选出合约作为主力合约时的行情数据
    # df4_1 = df4_1[(df4_1['calender_day'].between(value[0], value[1]))]

    # 1.1.3, 向量化计算收益 (速度快但细节处理有限)
    # 做多
    # df4_1['trade_signal'] = np.select(
    #     [
    #         (df4_1['start_time']==True)
    #         & (df4_1['ma20']>df4_1['ma40'])
    #         & ((df4_1['last_price']>df4_1['local_max']) & (df4_1['last_price']<=(df4_1['local_max']+3))),
    #         (df4_1['long_stop_profit']==True) | (df4_1['last_price']<df4_1['ma40'])
    #     ],
    #     [
    #         1,
    #         0
    #     ],
    #     default=np.nan)
    # 做空
    # df4_1['trade_signal'] = np.select(
    #    [
    #        (df4_1['start_time']==True)
    #        & (df4_1['ma20']<df4_1['ma40'])
    #        & ((df4_1['last_price']<df4_1['local_min']) & (df4_1['last_price']>=(df4_1['local_min']-3))),
    #        (df4_1['short_stop_profit']==True) | (df4_1['last_price']>df4_1['ma40'])
    #    ],
    #    [
    #        1,
    #        0
    #    ],
    #    default=np.nan)
    # result1 = cumulative_prod_cal_return(df4_1)
    # print(key, sum(result1))

    # 1.1.3, 遍历计算收益 (速度较慢, 但收益更接近真实情形)
    # long_position = False # 是否有持仓
    # long_open_cost = 0 # 持仓成本
    # long_stop_price1 = 0 # 止损价
    # long_stop_price2 = 0 # 止盈价
    # long_start_open = False # 是否可以建仓
    # short_position = False
    # short_open_cost = 0
    # short_stop_price1 = 0
    # short_stop_price2 = 0
    # short_start_open = False
    # now_local_min = 0 # 当前低点
    # pre_local_min = 0 # 上一个低点
    # pre_pre_local_min = 0 # 上上一个低点
    # now_local_max = 0
    # pre_local_max = 0
    # pre_pre_local_max = 0
    # profit_list = []
    # trade_signals = []
    # for row in df4_1.itertuples(index=False):
    #     last_price = row.last_price
    #     ma5 = row.ma5
    #     ma10 = row.ma10
    #     ma20 = row.ma20
    #     ma40 = row.ma40
    #     ma60 = row.ma60
    #     local_min = row.local_min
    #     local_max = row.local_max
    #     long_stop_profit = row.long_stop_profit
    #     short_stop_profit = row.short_stop_profit
    #     atr = row.atr
    #     start_time = row.start_time

    #     # 做多
    #     if long_position==False and start_time==True and long_start_open==True \
    #         and ma20>ma40 \
    #         and (now_local_max+2)>=last_price>now_local_max:
    #         long_position = True
    #         long_open_cost = last_price
    #         long_stop_price1 = last_price - atr
    #         long_start_open = False
    #         trade_signals.append(1)
    #     elif long_position==True \
    #         and (long_stop_profit==True or last_price<long_stop_price1 or last_price<ma40 or last_price<now_local_min):
    #         long_position = False
    #         profit_list.append(last_price/long_open_cost - 1)
    #         trade_signals.append(0)
    #     else:
    #         trade_signals.append(np.nan)
    #     # 做空
    #     if short_position==False and start_time==True and short_start_open==True \
    #         and ma20<ma40 \
    #         and now_local_min>last_price>=(now_local_min-2):
    #         short_position = True
    #         short_open_cost = last_price
    #         short_stop_price1 = last_price + atr
    #         short_start_open = False
    #         trade_signals.append(1)
    #     elif short_position==True \
    #         and (short_stop_profit==True or last_price>short_stop_price1 or last_price>ma40 or last_price>now_local_max):
    #         short_position = False
    #         profit_list.append(short_open_cost/last_price - 1)
    #         trade_signals.append(0)
    #     else:
    #        trade_signals.append(np.nan)

    #     # 更新交易信号
    #     if now_local_min != local_min:
    #         pre_pre_local_min = pre_local_min
    #         pre_local_min = now_local_min
    #         now_local_min = local_min
    #         long_start_open = True
    #     if now_local_max != local_max:
    #         pre_pre_local_max = pre_local_max
    #         pre_local_max = now_local_max
    #         now_local_max = local_max
    #         short_start_open = True

    # print(key, sum(profit_list), len(profit_list))

    # 1.1.4, 创建实时更新的动态K线回测图表
    # df4_2 = df4_1.copy()
    # df4_2['trade_signal'] = trade_signals
    # df4_2_1 = df4_2[df4_2['trade_signal'] == 1].copy()
    # df4_2_1.rename(columns={'last_price': 'open_signal'}, inplace=True)
    # df4_2_1['update_time'] = df4_2_1['update_time'] - pd.to_timedelta(df4_2_1['update_time'].dt.seconds % (15 * 60), unit='s')
    # df4_2_1['datetime'] = df4_2_1['calender_day'] + df4_2_1['update_time']
    # df4_2_2 = df4_2[df4_2['trade_signal'] == 0].copy()
    # df4_2_2.rename(columns={'last_price': 'close_signal'}, inplace=True)
    # df4_2_2['update_time'] = df4_2_2['update_time'] - pd.to_timedelta(df4_2_2['update_time'].dt.seconds % (15 * 60), unit='s')
    # df4_2_2['datetime'] = df4_2_2['calender_day'] + df4_2_2['update_time']
    # df4_2 = pd.merge(df4_2_1, df4_2_2, on=['datetime'], how='outer')
    # df4_2 = df4_2[['datetime', 'open_signal', 'close_signal']]
    # df3_2 = df3_1[(df3_1['datetime'].between(value[0], value[1]))]
    # df3_2 = df3_2[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'ma5', 'ma10', 'ma20', 'ma40', 'ma60', 'local_min', 'local_max', 'spread']]
    # df5_1 = pd.merge(df3_2, df4_2, on=['datetime'], how='outer')
    # df5_1.set_index('datetime', inplace=True)
    # if len(df5_1) > 0:
    #     custom_style = mpf.make_mpf_style(
    #             base_mpf_style='classic', # 基础样式
    #             marketcolors=mpf.make_marketcolors(up='red', down='green', # 上涨和下跌的颜色
    #                                                 wick={'up': 'red', 'down': 'green'}, # 烛芯的颜色
    #                                                 edge={'up': 'red', 'down': 'green'}, # 边缘的颜色
    #                                                 volume={'up': 'red', 'down': 'green'}), # 成交量的颜色
    #             gridcolor='gray', # 网格线的颜色
    #             gridstyle='--', # 网格线的样式
    #             facecolor='white', # 背景颜色
    #             figcolor='white', # 图表颜色
    #             edgecolor='white', # 边缘颜色
    #             rc={'font.family': font.get_name(), 'axes.unicode_minus': False}) # 字体格式
    #     fig = mpf.figure(figsize=(13, 10))
    #     gs = fig.add_gridspec(5, 1)
    #     ax1 = fig.add_subplot(gs[0:3, 0])
    #     ax2 = fig.add_subplot(gs[3:4, 0], sharex=ax1)
    #     ax3 = fig.add_subplot(gs[4:5, 0], sharex=ax1)
    #     def animate(i):
    #         ax1.clear()
    #         ax2.clear()
    #         ax3.clear()

    #         start = max(0, min(len(df5_1), i + 1) - 100)
    #         data = df5_1.iloc[start : i + 1]
    #         local_min_list = list(data['local_min'])[4:] + [np.nan]*4
    #         local_max_list = list(data['local_max'])[4:] + [np.nan]*4
    #         spread_list = list(data['spread'])
    #         open_signal_list = list(data['open_signal'])
    #         close_signal_list = list(data['close_signal'])
    #         ma5_list = list(data['ma5'])
    #         ma10_list = list(data['ma10'])
    #         ma20_list = list(data['ma20'])
    #         ma40_list = list(data['ma40'])
    #         ma60_list = list(data['ma60'])
    #         volume_list = list(data['volume'])
    #         open_interest_list = list(data['open_interest'])

    #         # 方式一: 使用make_addplot添加买卖信号
    #         # ap1 = mpf.make_addplot(open_signal_list, type='scatter', ax=ax1, marker='^', markersize=20, color='#000000')
    #         # ap2 = mpf.make_addplot(close_signal_list, type='scatter', ax=ax1, marker='v', markersize=20, color='#000000')
    #         # mpf.plot(data, type='candle', style=custom_style, ax=ax1, volume=ax2, addplot=[ap1, ap2])

    #         # 方式二: 直接在ax1上绘制买卖信号
    #         mpf.plot(data, type='candle', style=custom_style, ax=ax1, volume=False)
    #         ax1.plot(local_min_list, 'o', markersize=3, color='#000000', label='local_min')
    #         ax1.plot(local_max_list, 'o', markersize=3, color='#000000', label='local_max')
    #         # ax1.text(0.5, 0.9, 'spread: ' + str(spread_list[-1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='black')
    #         ax1.plot(open_signal_list, '^', markersize=6, color='#000000', label='open_signal')
    #         ax1.plot(close_signal_list, 'v', markersize=6, color='#000000', label='close_signal')
    #         # ax1.plot(ma5_list, color='red', linewidth=1)
    #         # ax1.plot(ma10_list, color='yellow', linewidth=1)
    #         ax1.plot(ma20_list, color='green', linewidth=1)
    #         ax1.plot(ma40_list, color='blue', linewidth=1)
    #         # ax1.plot(ma60_list, color='purple', linewidth=1)
    #         ax2.bar(range(len(volume_list)), volume_list, color='#DEB887', label='Volume')
    #         ax3.bar(range(len(open_interest_list)), open_interest_list, color='#DEB887', label='OI')

    #         plt.setp(ax1.get_xticklabels(), visible=False)
    #         plt.setp(ax2.get_xticklabels(), visible=False)
    #         plt.tight_layout()

    #     ani = FuncAnimation(fig, animate, frames=range(len(df5_1)), interval=50, repeat=False)
    #     mpf.show()
    #     plt.close('all')

    # 1.2, 策略二: 1分钟放量反向进场, 固定止盈, 因为放量点位在不停地变化, 所以止损和止盈同逻辑 (价格与止盈点数的比例按1000: 1) ——>1min, 15min均可行, 其中1min收益率高, 15min收益率低
    # 1.2.1, 处理tick数据
    # df4_1 = pd.merge(df2_2, df3_1, left_on=['datetime'], right_on=['datetime'], how='outer')
    # df4_1['ma5'] = df4_1['ma5'].ffill()
    # df4_1['ma10'] = df4_1['ma10'].ffill()
    # df4_1['ma20'] = df4_1['ma20'].ffill()
    # df4_1['ma40'] = df4_1['ma40'].ffill()
    # df4_1['ma60'] = df4_1['ma60'].ffill()
    # df4_1['start_time'] = ~(((df4_1['update_time']>=pd.to_timedelta('20:55:00')) & (df4_1['update_time']<pd.to_timedelta('21:15:00')))
    #                         | ((df4_1['update_time']>=pd.to_timedelta('14:45:00')) & (df4_1['update_time']<=pd.to_timedelta('15:00:00')))) # 限制开仓时间段

    # 1.2.2, 筛选出合约作为主力合约时的行情数据
    # df4_1 = df4_1[(df4_1['calender_day'].between(value[0], value[1]))]

    # 1.2.3, 遍历计算收益 (速度较慢, 但收益更接近真实情形)
    # long_position = False # 是否有持仓
    # long_open_cost = 0 # 持仓成本
    # long_stop_price1 = 0 # 止损价
    # long_stop_price2 = 0 # 止盈价
    # long_start_open = False # 是否可以建仓
    # short_position = False
    # short_open_cost = 0
    # short_stop_price1 = 0
    # short_stop_price2 = 0
    # short_start_open = False
    # now_local_min = 0 # 当前低点
    # pre_local_min = 0 # 上一个低点
    # pre_pre_local_min = 0 # 上上一个低点
    # now_local_max = 0
    # pre_local_max = 0
    # pre_pre_local_max = 0
    # profit_list = []
    # trade_signals = []
    # for row in df4_1.itertuples(index=False):
    #     last_price = row.last_price
    #     long_stop_profit = row.long_stop_profit
    #     short_stop_profit = row.short_stop_profit
    #     start_time = row.start_time

    #     # 做多
    #     if long_position==False and start_time==True and long_start_open==True \
    #         and short_stop_profit==True:
    #         long_position = True
    #         long_open_cost = last_price
    #         long_stop_price1 = last_price + 6
    #         long_start_open = False
    #         trade_signals.append(1)
    #     elif long_position==True \
    #         and (long_stop_profit==True or last_price>=long_stop_price1):
    #         long_position = False
    #         profit_list.append(last_price/long_open_cost - 1)
    #         trade_signals.append(0)
    #     else:
    #         trade_signals.append(np.nan)
    #     # 做空
    #     if short_position==False and start_time==True and short_start_open==True \
    #         and long_stop_profit==True:
    #         short_position = True
    #         short_open_cost = last_price
    #         short_stop_price1 = last_price - 6
    #         short_start_open = False
    #         trade_signals.append(1)
    #     elif short_position==True \
    #         and (short_stop_profit==True or last_price<=short_stop_price1):
    #         short_position = False
    #         profit_list.append(short_open_cost/last_price - 1)
    #         trade_signals.append(0)
    #     else:
    #        trade_signals.append(np.nan)

    #     # 更新交易信号
    #     if now_local_min != local_min:
    #         pre_pre_local_min = pre_local_min
    #         pre_local_min = now_local_min
    #         now_local_min = local_min
    #         long_start_open = True
    #     if now_local_max != local_max:
    #         pre_pre_local_max = pre_local_max
    #         pre_local_max = now_local_max
    #         now_local_max = local_max
    #         short_start_open = True

    # print(key, sum(profit_list), len(profit_list))

    # 1.3, 策略三: 用上一个交易日的最高最低价作为高低点, 然后在临近高低点附近开仓
    # 1.3.1, 处理tick数据
    # df4_1 = pd.merge(df2_2, df3_1, left_on=['datetime'], right_on=['datetime'], how='outer')
    # df4_1['ma5'] = df4_1['ma5'].ffill()
    # df4_1['ma10'] = df4_1['ma10'].ffill()
    # df4_1['ma20'] = df4_1['ma20'].ffill()
    # df4_1['ma40'] = df4_1['ma40'].ffill()
    # df4_1['ma60'] = df4_1['ma60'].ffill()
    # df4_1['atr'] = df4_1['atr'].ffill()
    # df4_1['yd_low'] = df4_1['yd_low'].ffill()
    # df4_1['yd_high'] = df4_1['yd_high'].ffill()
    # df4_1['start_time'] = ~((df4_1['update_time']>=pd.to_timedelta('14:45:00')) & (df4_1['update_time']<=pd.to_timedelta('15:00:00'))) # 限制开仓时间段

    # 1.3.2, 筛选出合约作为主力合约时的行情数据
    # df4_1 = df4_1[(df4_1['calender_day'].between(value[0], value[1]))]

    # 1.3.3, 遍历计算收益 (速度较慢, 但收益更接近真实情形)
    # long_position = False # 是否有持仓
    # long_open_cost = 0 # 持仓成本
    # long_stop_price1 = 0 # 止损价
    # long_stop_price2 = 0 # 止盈价
    # long_start_open = True # 是否可以建仓
    # short_position = False
    # short_open_cost = 0
    # short_stop_price1 = 0
    # short_stop_price2 = 0
    # short_start_open = True
    # pre_yd_low = 0
    # pre_yd_high = 0
    # profit_list = []
    # trade_signals = []
    # for row in df4_1.itertuples(index=False):
    #     last_price = row.last_price
    #     ma5 = row.ma5
    #     ma10 = row.ma10
    #     ma20 = row.ma20
    #     ma40 = row.ma40
    #     ma60 = row.ma60
    #     atr = int(np.nan_to_num(row.atr, nan=8.0))
    #     yd_low = row.yd_low
    #     yd_high = row.yd_high
    #     start_time = row.start_time

    #     # 做多
    #     if long_position==False and start_time==True and long_start_open==True \
    #         and ma10>ma20>ma40 \
    #         and (yd_high-atr+1)>=last_price>=(yd_high-atr):
    #         long_position = True
    #         long_open_cost = last_price
    #         long_stop_price1 = last_price - atr
    #         long_stop_price2 = yd_high
    #         long_start_open = False
    #         trade_signals.append(1)
    #     elif long_position==True \
    #         and (last_price<long_stop_price1 or last_price>=long_stop_price2):
    #         long_position = False
    #         # long_start_open = True
    #         profit_list.append(last_price/long_open_cost - 1)
    #         trade_signals.append(0)
    #     else:
    #         trade_signals.append(np.nan)
    #     # 做空
    #     if short_position==False and start_time==True and short_start_open==True \
    #         and ma10<ma20<ma40 \
    #         and (yd_low+atr)>=last_price>=(yd_low+atr-1):
    #         short_position = True
    #         short_open_cost = last_price
    #         short_stop_price1 = last_price + atr
    #         short_stop_price2 = yd_low
    #         short_start_open = False
    #         trade_signals.append(1)
    #     elif short_position==True \
    #         and (last_price>short_stop_price1 or last_price<=short_stop_price2):
    #         short_position = False
    #         # short_start_open = True
    #         profit_list.append(short_open_cost/last_price - 1)
    #         trade_signals.append(0)
    #     else:
    #        trade_signals.append(np.nan)

    #     # 更新交易信号
    #     if yd_low != pre_yd_low:
    #         pre_yd_low = yd_low
    #         short_start_open = True
    #     if yd_high != pre_yd_high:
    #         pre_yd_high = yd_high
    #         long_start_open = True

    # print(key, sum(profit_list), len(profit_list))

    # 1.3.4, 创建实时更新的动态K线回测图表
    # df4_2 = df4_1.copy()
    # df4_2['trade_signal'] = trade_signals
    # df4_2_1 = df4_2[df4_2['trade_signal'] == 1].copy()
    # df4_2_1.rename(columns={'last_price': 'open_signal'}, inplace=True)
    # df4_2_1['update_time'] = df4_2_1['update_time'] - pd.to_timedelta(df4_2_1['update_time'].dt.seconds % (15 * 60), unit='s')
    # df4_2_1['datetime'] = df4_2_1['calender_day'] + df4_2_1['update_time']
    # df4_2_2 = df4_2[df4_2['trade_signal'] == 0].copy()
    # df4_2_2.rename(columns={'last_price': 'close_signal'}, inplace=True)
    # df4_2_2['update_time'] = df4_2_2['update_time'] - pd.to_timedelta(df4_2_2['update_time'].dt.seconds % (15 * 60), unit='s')
    # df4_2_2['datetime'] = df4_2_2['calender_day'] + df4_2_2['update_time']
    # df4_2 = pd.merge(df4_2_1, df4_2_2, on=['datetime'], how='outer')
    # df4_2 = df4_2[['datetime', 'open_signal', 'close_signal']]
    # df3_2 = df3_1[(df3_1['datetime'].between(value[0], value[1]))]
    # df3_2 = df3_2[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'yd_high', 'yd_low']]
    # df5_1 = pd.merge(df3_2, df4_2, on=['datetime'], how='outer')
    # df5_1.set_index('datetime', inplace=True)
    # if len(df5_1) > 0:
    #     custom_style = mpf.make_mpf_style(
    #             base_mpf_style='classic', # 基础样式
    #             marketcolors=mpf.make_marketcolors(up='red', down='green', # 上涨和下跌的颜色
    #                                                 wick={'up': 'red', 'down': 'green'}, # 烛芯的颜色
    #                                                 edge={'up': 'red', 'down': 'green'}, # 边缘的颜色
    #                                                 volume={'up': 'red', 'down': 'green'}), # 成交量的颜色
    #             gridcolor='gray', # 网格线的颜色
    #             gridstyle='--', # 网格线的样式
    #             facecolor='white', # 背景颜色
    #             figcolor='white', # 图表颜色
    #             edgecolor='white', # 边缘颜色
    #             rc={'font.family': font.get_name(), 'axes.unicode_minus': False}) # 字体格式
    #     fig = mpf.figure(figsize=(13, 10))
    #     gs = fig.add_gridspec(5, 1)
    #     ax1 = fig.add_subplot(gs[0:3, 0])
    #     ax2 = fig.add_subplot(gs[3:4, 0], sharex=ax1)
    #     ax3 = fig.add_subplot(gs[4:5, 0], sharex=ax1)
    #     def animate(i):
    #         ax1.clear()
    #         ax2.clear()
    #         ax3.clear()

    #         start = max(0, min(len(df5_1), i + 1) - 100)
    #         data = df5_1.iloc[start : i + 1]
    #         yd_high_list = list(data['yd_high'])
    #         yd_low_list = list(data['yd_low'])
    #         open_signal_list = list(data['open_signal'])
    #         close_signal_list = list(data['close_signal'])
    #         volume_list = list(data['volume'])
    #         open_interest_list = list(data['open_interest'])

    #         mpf.plot(data, type='candle', style=custom_style, ax=ax1, volume=False)
    #         ax1.plot(yd_high_list, '-', markersize=1, color='#000000', label='yd_high')
    #         ax1.plot(yd_low_list, '-', markersize=1, color='#000000', label='yd_low')
    #         ax1.text(0.5, 0.9, 'yd_high:' + str(yd_high_list[-1]) + ' yd_low:' + str(yd_low_list[-1]), fontsize=15, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, color='black')
    #         ax1.plot(open_signal_list, '^', markersize=6, color='#000000', label='open_signal')
    #         ax1.plot(close_signal_list, 'v', markersize=6, color='#000000', label='close_signal')
    #         ax2.bar(range(len(volume_list)), volume_list, color='#DEB887', label='Volume')
    #         ax3.bar(range(len(open_interest_list)), open_interest_list, color='#DEB887', label='OI')

    #         plt.setp(ax1.get_xticklabels(), visible=False)
    #         plt.setp(ax2.get_xticklabels(), visible=False)
    #         plt.tight_layout()

    #     ani = FuncAnimation(fig, animate, frames=range(len(df5_1)), interval=50, repeat=False)
    #     mpf.show()
    #     plt.close('all')

    # 1.4, 策略四: 在vwap均线之上开多, 1个ATR止损, 在vwap均线之下开空, 1个ATR止损
    # 1.4.1, 处理tick数据
    # df4_1 = pd.merge(df2_2, df3_1, left_on=['datetime'], right_on=['datetime'], how='outer')
    # df4_1['atr'] = df4_1['atr'].ffill()
    # df4_1['vwap20'] = df4_1['vwap20'].ffill()
    # df4_1['start_time'] = ~((df4_1['update_time']>=pd.to_timedelta('14:45:00')) & (df4_1['update_time']<=pd.to_timedelta('15:00:00'))) # 限制开仓时间段

    # 1.4.2, 筛选出合约作为主力合约时的行情数据
    # df4_1 = df4_1[(df4_1['calender_day'].between(value[0], value[1]))]

    # 1.4.3, 遍历计算收益 (速度较慢, 但收益更接近真实情形)
    # long_position = False # 是否有持仓
    # long_open_cost = 0 # 持仓成本
    # long_stop_price1 = 0 # 止损价
    # long_stop_price2 = 0 # 止盈价
    # short_position = False
    # short_open_cost = 0
    # short_stop_price1 = 0
    # short_stop_price2 = 0
    # profit_list = []
    # trade_signals = []
    # for row in df4_1.itertuples(index=False):
    #     last_price = row.last_price
    #     atr = row.atr
    #     long_stop_profit = row.long_stop_profit
    #     short_stop_profit = row.short_stop_profit
    #     vwap20 = row.vwap20
    #     start_time = row.start_time

    #     # 做多
    #     if long_position==False and start_time==True \
    #         and (vwap20+1)>=last_price>=(vwap20):
    #         long_position = True
    #         long_open_cost = last_price
    #         long_stop_price1 = last_price - atr
    #         trade_signals.append(1)
    #     elif long_position==True \
    #         and (last_price<long_stop_price1 or long_stop_profit==True):
    #         long_position = False
    #         profit_list.append(last_price/long_open_cost - 1)
    #         trade_signals.append(0)
    #     else:
    #         trade_signals.append(np.nan)
    #     # 做空
    #     if short_position==False and start_time==True \
    #         and (vwap20)>last_price>=(vwap20-1):
    #         short_position = True
    #         short_open_cost = last_price
    #         short_stop_price1 = last_price + atr
    #         trade_signals.append(1)
    #     elif short_position==True \
    #         and (last_price>short_stop_price1 or short_stop_profit==True):
    #         short_position = False
    #         profit_list.append(short_open_cost/last_price - 1)
    #         trade_signals.append(0)
    #     else:
    #        trade_signals.append(np.nan)

    # print(key, sum(profit_list), len(profit_list))

    # 2, 遍历回测
    # def generate_rows(df):
    #     for idx, row in df.iterrows():
    #         yield idx, row
    # generate = generate_rows(df2_2)
    # 2.1, 版本一: 生成实时K线进行回测 (虽然更接近真实行情, 但耗时太长, 对电脑配置要求太高)
    # last_trade__day_begin = pd.to_timedelta('12:00:00')
    # last_trade_day_end = pd.to_timedelta('15:30:00')
    # now_trade_day_begin = pd.to_timedelta('20:30:00')
    # now_trade_day_end = pd.to_timedelta('12:00:00')
    # pre_update_time = None
    # gen = K_line_generator(15)
    # K_df = pd.DataFrame() # 历史K线数据, 只有上一个交易日及之前的K线数据
    # for row in df2_2.itertuples(index=False):
    #     if pre_update_time == None:
    #         gen = K_line_generator(15)
    #     elif (((pre_update_time>last_trade__day_begin) and (pre_update_time<=last_trade_day_end))
    #           and ((row.update_time>=now_trade_day_begin) or (row.update_time<now_trade_day_end))):
    #         gen.flush()
    #         temp_df = pd.DataFrame(gen.get_klines()) # 新的交易日将要开始, 取出上一个交易日的K线数据
    #         K_df = pd.concat([K_df, temp_df]) # 更新历史K线数据
    #         gen = K_line_generator(15)
    #     pre_update_time = row.update_time
    #     tick_time = row.datetime.timestamp()
    #     price = row.last_price
    #     cumulative_volume = row.volume
    #     gen.add_tick(tick_time, price, cumulative_volume)
    #     real_time_df = pd.concat([K_df, pd.DataFrame(gen.get_klines())]) # 实时K线数据, 只有上一根K线及之前的K线数据
    #     new_row = pd.DataFrame({'open_time': pd.Timestamp(gen.current_kline.open_time, unit='s'), 'open': gen.current_kline.open,
    #                             'high': gen.current_kline.high, 'low': gen.current_kline.low,
    #                             'close': gen.current_kline.close, 'volume': gen.current_kline.volume}, index=[0])
    #     real_time_df = pd.concat([real_time_df, new_row], ignore_index=True) # 实时K线数据, 包含最新的K线数据

    # 2.1, 版本二: 使用生产者消费者模式生成实时K线进行回测并创建实时更新的动态K线回测图表
    # last_trade_day_begin = pd.to_timedelta('12:00:00')
    # last_trade_day_end = pd.to_timedelta('15:30:00')
    # now_trade_day_begin = pd.to_timedelta('20:30:00')
    # now_trade_day_end = pd.to_timedelta('12:00:00')
    # pre_update_time = None
    # gen = K_line_generator(1)
    # K_df = pd.DataFrame()
    # window_df_queue = queue.Queue()
    # df_lock = threading.Lock()
    # def update_window_df():
    #     global pre_update_time, gen, K_df, window_df_queue
    #     for row in df2_2.itertuples(index=False):
    #         if pre_update_time is None:
    #             gen = K_line_generator(1)
    #         elif (((pre_update_time > last_trade_day_begin) and (pre_update_time <= last_trade_day_end))
    #                 and ((row.update_time >= now_trade_day_begin) or (row.update_time < now_trade_day_end))):
    #             gen.flush()
    #             temp_df = pd.DataFrame(gen.get_klines())
    #             K_df = pd.concat([K_df, temp_df])
    #             gen = K_line_generator(1)
    #         pre_update_time = row.update_time
    #         tick_time = row.datetime.timestamp()
    #         price = row.last_price
    #         cumulative_volume = row.volume
    #         gen.add_tick(tick_time, price, cumulative_volume)
    #         real_time_df = pd.concat([K_df, pd.DataFrame(gen.get_klines())]).copy() # 实时K线数据, 只有上一根K线及之前的K线数据
    #         new_row = pd.DataFrame({'open_time': pd.Timestamp(gen.current_kline.open_time, unit='s'), 'open': gen.current_kline.open,
    #                                 'high': gen.current_kline.high, 'low': gen.current_kline.low,
    #                                 'close': gen.current_kline.close, 'volume': gen.current_kline.volume}, index=[0])
    #         real_time_df = pd.concat([real_time_df, new_row], ignore_index=True) # 实时K线数据, 包含最新的K线数据
    #         window_df = real_time_df.tail(100).copy()
    #         window_df.set_index('open_time', inplace=True)

    #         df_lock.acquire()
    #         try:
    #             window_df_queue.put(window_df, block=False)
    #         except queue.Full:
    #             print('队列已满, 无法添加新的数据')
    #         except Exception as e:
    #             print(f'发生错误: {e}')
    #         finally:
    #             df_lock.release()

    # custom_style = mpf.make_mpf_style(
    #     base_mpf_style='classic',
    #     marketcolors=mpf.make_marketcolors(up='red', down='green',
    #                                     wick={'up': 'red', 'down': 'green'},
    #                                     edge={'up': 'red', 'down': 'green'},
    #                                     volume={'up': 'red', 'down': 'green'}),
    #     gridcolor='gray',
    #     gridstyle='--',
    #     facecolor='white',
    #     figcolor='white',
    #     edgecolor='white',
    #     rc={'font.family': font.get_name(), 'axes.unicode_minus': False}
    # )
    # fig = mpf.figure(figsize=(13, 10))
    # gs = fig.add_gridspec(5, 1)
    # ax1 = fig.add_subplot(gs[0:3, 0])
    # ax2 = fig.add_subplot(gs[3:5, 0], sharex=ax1)

    # def animate(frame):
    #     global window_df_queue
    #     df_lock.acquire()
    #     if not window_df_queue.empty():
    #         ax1.clear()
    #         ax2.clear()
    #         mpf.plot(window_df_queue.get(), type='candle', style=custom_style, ax=ax1, volume=ax2)
    #         plt.setp(ax1.get_xticklabels(), visible=False)
    #         plt.tight_layout()
    #     df_lock.release()

    # producer_thread = threading.Thread(target=update_window_df)
    # producer_thread.daemon = True
    # producer_thread.start()
    # producer_thread.join()

    # ani = FuncAnimation(fig, animate, interval=50, blit=False)
    # mpf.show()
    # plt.close('all')
