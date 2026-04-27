# 该项目的作用: 观察策略的进出场信号表现
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

    # 1, 处理K线数据，计算交易信号
    # 处理高频数据
    df3_1 = MQL.StaticTickToK.tick_to_K(df2_2, "5min")

    df3_1["rsi"] = RSI(df3_1["close"], 20)
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

    df3_1.loc[df3_1["rsi"] <= df3_1["rsi_0.05_percentile"], "trade_signal"] = 1
    # df3_1.loc[df3_1["rsi"] >= df3_1["rsi_0.95_percentile"], "trade_signal"] = 1

    df3_1 = df3_1[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "rsi",
            "trade_signal",
        ]
    ]

    # 处理中频数据
    df3_2 = MQL.StaticTickToK.tick_to_K(df2_2, "1h")

    df3_2["ma20"] = MA(df3_2["close"], 20)
    df3_2["ma60"] = MA(df3_2["close"], 60)

    df3_2.loc[df3_2["ma20"] > df3_2["ma60"], "trade_signal"] = 1
    # df3_2.loc[df3_2["ma20"] < df3_2["ma60"], "trade_signal"] = 1

    df3_2 = df3_2[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "ma20",
            "ma60",
            "trade_signal",
        ]
    ]

    # 处理低频数据
    df3_3 = MQL.StaticTickToK.tick_to_K(df2_2, "1D")

    df3_3["ma5"] = MA(df3_3["close"], 5)
    df3_3["ma20"] = MA(df3_3["close"], 20)

    df3_3.loc[df3_3["ma5"] > df3_3["ma20"], "trade_signal"] = 1
    # df3_3.loc[df3_3["ma5"] < df3_3["ma20"], "trade_signal"] = 1

    df3_3 = df3_3[
        [
            "trading_day",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "ma5",
            "ma20",
            "trade_signal",
        ]
    ]
    df3_3.rename(columns={"trading_day": "datetime"}, inplace=True)

    # 2, 筛选出合约作为主力合约时的行情数据
    df3_1 = df3_1[(df3_1["datetime"].between(value[0], value[1]))]
    df3_2 = df3_2[(df3_2["datetime"].between(value[0], value[1]))]
    df3_3 = df3_3[(df3_3["datetime"].between(value[0], value[1]))]

    # 3, 需要观察得信号
    plot = MQL.KLineAnalyzer()
    additional_plots_configs = [
        # 高频配置: 画成交量和rsi
        {
            "volume": {"type": "bar", "panel": 1, "color": "gray"},
            "rsi": {"type": "line", "panel": 2, "color": "purple"},
        },
        # 中频配置: 画持仓量
        {
            "open_interest": {"type": "bar", "panel": 1, "color": "gray"},
        },
        # 低频配置: 画ma5和ma20
        {
            "ma5": {"type": "line", "panel": 0, "color": "blue"},
            "ma20": {"type": "line", "panel": 0, "color": "red"},
        },
    ]
    plot.analyze_trade_signals(
        [df3_1, df3_2, df3_3],
        key,
        [30, 20, 10],
        [30, 20, 10],
        "trade_signal",
        additional_plots_configs,
    )
