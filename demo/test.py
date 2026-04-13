# 该项目的作用: 实现期货策略tick级别的回测
import pandas as pd
import numpy as np
import queue
import threading
import bisect

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.animation import FuncAnimation

import sys

sys.path.append("D:\\LearningAndWorking\\VSCode\\python\\module\\")
from MyTT import *

# 设置全局字体
fm.fontManager.addfont(
    "C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)
mpl.rcParams["font.sans-serif"] = "FZHei-B01S"
mpl.rcParams["axes.unicode_minus"] = False
font = fm.FontProperties(
    fname="C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)


# 1, 向量化回测所用类或函数
def find_per_value(array, percentile_value):
    """
    功能: 找出数组中处于某分位值的数

    参数:
        array(np.ndarray): 数组
        percentile_value(float): 分位值 (0-1)

    返回:
        float or NaN
    """
    array = np.asarray(array)

    if not 0 <= percentile_value <= 1:
        raise ValueError("percentile_value must be between 0 and 1")

    # 获取历史值 (排除当前值)
    historical_values = array[:-1]

    # 检查历史值是否为空
    if len(historical_values) == 0:
        return np.nan

    # 检查历史值是否全是NaN
    if np.all(np.isnan(historical_values)):
        return np.nan

    # 计算分位数
    return np.nanpercentile(historical_values, percentile_value * 100)


def find_rank_value(array):
    """
    功能: 确定当前值在历史数据中的百分位

    参数:
        array(np.ndarray): 数组

    返回:
        float or NaN
    """
    array = np.array(array)

    # 1. 基础长度检查
    if len(array) < 2:
        return np.nan

    current_value = array[-1]  # 当前值
    historical_values = array[:-1]  # 前面的历史值

    # 2. 检查历史数据是否为空
    if len(historical_values) == 0:
        return np.nan

    # 3. 如果历史数据全是NaN, 无法计算有意义的排名, 直接返回NaN
    if np.all(np.isnan(historical_values)):
        return np.nan

    # 4. 如果当前值本身是NaN, 也无法计算排名
    if np.isnan(current_value):
        return np.nan

    # 为了严谨, 只统计非NaN的历史值数量作为分母 (如果历史数据中有零星NaN)
    valid_historical_count = np.sum(~np.isnan(historical_values))

    if valid_historical_count == 0:
        return np.nan

    count_leq_current = np.sum(
        (historical_values <= current_value) & (~np.isnan(historical_values))
    )

    return count_leq_current / valid_historical_count


def find_local_extrema(series, window, location, extrema_type="both"):
    """
    功能: 局部极值检测函数, 可同时检测高点和低点

    参数:
        series(pd.Series/list/np.ndarray): 数据序列
        window(int): 窗口大小
        location(int): 需要判断极值的位置
        extrema_type(str): 'high','low','both'

    返回:
        float or NaN 或 dict
    """
    if not isinstance(series, (pd.Series, list, np.ndarray)):
        raise TypeError("series必须是pd.Series,list或np.ndarray类型")

    series = np.asarray(series)

    if len(series) != window:
        raise ValueError(f"输入必须是{window}长度的数据, 实际长度为{len(series)}")

    if not 1 <= location <= window:
        raise ValueError(f"location必须在1到{window}范围内, 实际值为{location}")

    target_value = series[location - 1]

    if extrema_type == "high":
        max_value = np.max(series)
        return target_value if np.isclose(target_value, max_value) else np.nan
    elif extrema_type == "low":
        min_value = np.min(series)
        return target_value if np.isclose(target_value, min_value) else np.nan
    elif extrema_type == "both":
        max_value = np.max(series)
        min_value = np.min(series)
        high_result = target_value if np.isclose(target_value, max_value) else np.nan
        low_result = target_value if np.isclose(target_value, min_value) else np.nan
        return {"high": high_result, "low": low_result}
    else:
        raise ValueError("extrema_type必须是'high','low',或'both'")


def propagate_true_down(df, col_name, window):
    """
    功能: df[col_name]由空值, bool值共同构成, 找到df[col_name]中为True的位置,
         将其所在行下面的window行也设为True (不管原来是不是空值)

    参数:
        df(pd.DataFrame): 数据框
        col_name(str): 列名
        window(int): 向下传播的行数 (不包含当前行)

    返回:
        df(pd.DataFrame)
    """
    df = df.copy()

    # 找出所有True的位置 (忽略NaN)
    true_indices = df.index[df[col_name] == True].tolist()
    for idx in true_indices:
        end_idx = idx + window
        end_idx = min(end_idx, df.index[-1])
        df.loc[idx:end_idx, col_name] = True

    return df


class Trigger_resampler:
    """
    功能: 基于 "触发机制" 的重采样. 只有当第N+1次变动发生时, 才将前N次变动打包.

    参数:
        df (pd.DataFrame): 有datetime, last_price, volume, open_interest等列
        time_col (str): 时间列名
        count_limit (int): 多少个变动区间 (以1分钟为单位) 构成一个桶
    """

    def __init__(self, df, time_col, count_limit):
        self.df = df.copy()
        self.time_col = time_col
        self.count_limit = count_limit

        # 1. 预处理与分桶逻辑 (在初始化时完成)
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        self.df.sort_values(self.time_col, inplace=True)

        # 计算分钟变动
        current_minute = self.df[self.time_col].dt.floor("min")
        shifted_minute = current_minute.shift(1)
        is_change = (current_minute != shifted_minute).fillna(True)

        # 计算变动计数并分桶
        change_count = is_change.cumsum()
        self.buckets = (change_count - 1) // self.count_limit

        # 创建内部的GroupBy对象
        self._grouped = self.df.groupby(self.buckets)

    def agg(self, func_dict):
        """
        执行聚合操作
        """
        # 2. 执行聚合
        # 调用内部groupby对象的agg方法
        result = self._grouped.agg(func_dict)

        return result


# 封装一个便捷函数，方便调用
def trigger_based_resample(df, time_col, count_limit):
    """
    工厂函数: 返回一个Trigger_resampler对象
    """
    return Trigger_resampler(df, time_col, count_limit)


def tick_to_K(df, time):
    """
    功能: tick数据转K线数据 (支持分钟/小时/日级别K线)

    注意: resample函数是一个基于物理时间的分组工具, 如果不特意设置, 起始时间默认为00:00.
        对于1小时K线合成, 如果第一个数据出现在10:02, 且10:00-10:15之间处于停盘时间, 那么
        resample会将10:00-11:00的数据合成一根K线.

    核心逻辑:
        1, 基于"没有日盘一定不会有夜盘, 但没有夜盘不一定没有日盘"的逻辑先构建交易日序列
        2, 识别日盘 (09:00-15:00) 的日期, 如果日历日是D, 则交易日是D
        3, 识别夜盘 (21:00-23:59) 的日期, 如果日历日是D, 则交易日是D之后的第一个交易日 (从交易日序列中找)
        4, 识别夜盘 (00:00-06:00) 的日期, 如果日历日是D+1, 则交易日是D之后的第一个交易日 (从交易日序列中找)
        5, 分组重采样

    参数:
        df(pd.DataFrame): tick行情数据, 有datetime, instrument_id, last_price, volume, open_interest等列
        time(str): K线周期, 如'5min', '1h', '1D'等

    返回:
        df(pd.DataFrame): 有datetime (日K用trading_day), instrument_id, open, high, low, close, volume, open_interest等列
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "instrument_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )

    df = df.copy()

    # 1, 日期时间预处理
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "instrument_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )

    # 检查必需列是否存在
    required_columns = [
        "datetime",
        "instrument_id",
        "last_price",
        "volume",
        "open_interest",
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    s_date = df["datetime"].dt.date
    s_time = df["datetime"].dt.time

    # 2, 识别有效交易日序列
    t_day_start = pd.Timestamp("09:00").time()
    t_day_end = pd.Timestamp("15:00").time()
    mask_day_session = (s_time >= t_day_start) & (s_time <= t_day_end)
    valid_trading_days_sorted = sorted(s_date.loc[mask_day_session].unique())

    if not valid_trading_days_sorted:
        return pd.DataFrame(
            columns=[
                "datetime",
                "instrument_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )

    valid_trading_days_set = set(valid_trading_days_sorted)
    vtd_list = valid_trading_days_sorted
    len_vtd = len(vtd_list)

    # =================================================================
    # [性能优化1]交易日映射向量化 (替代Python for循环)
    # =================================================================
    unique_calendar_days = sorted(s_date.unique())
    date_map = {}
    # 在唯一日期列表上循环
    for d in unique_calendar_days:
        if d in valid_trading_days_set:
            date_map[d] = d
        else:
            idx = bisect.bisect_right(vtd_list, d)
            date_map[d] = (
                vtd_list[idx] if idx < len_vtd else vtd_list[-1] if vtd_list else None
            )  # 当bisect找不到有效日期时返回最后一个交易日

    s_next_trade_day = s_date.map(date_map)  # 向量化应用映射 (核心加速点)

    # 3, 分配具体的交易日期 (_trade_date)
    trade_dates = pd.Series(pd.NaT, index=df.index, dtype="object")
    t_night_start = pd.Timestamp("21:00").time()
    t_midnight_end = pd.Timestamp("23:59:59").time()
    t_night_end = pd.Timestamp("06:00").time()

    # ---日盘---
    # 日盘时段 (09:00-15:00) ->交易日=日历日
    mask_day = (s_time >= t_day_start) & (s_time <= t_day_end)
    valid_day_mask = mask_day & s_date.isin(valid_trading_days_set)
    trade_dates.loc[valid_day_mask] = s_date.loc[valid_day_mask]

    # ---夜盘前段 (21:00-23:59) ---
    # 21:00-23:59->交易日=日历日+1 (如果该日是交易日) , 否则找下一个交易日
    mask_night1 = (s_time >= t_night_start) & (s_time <= t_midnight_end)
    valid_n1_mask = mask_night1 & s_next_trade_day.notna()
    trade_dates.loc[valid_n1_mask] = s_next_trade_day.loc[
        valid_n1_mask
    ]  # 直接使用预计算好的s_next_trade_day, 无需再次查找

    # ---夜盘后段 (00:00 - 06:00) ---
    # 00:00-06:00->交易日=日历日 (如果该日是交易日) , 否则找下一个交易日
    mask_night2 = s_time <= t_night_end
    is_today_valid = mask_night2 & s_date.isin(valid_trading_days_set)
    trade_dates.loc[is_today_valid] = s_date.loc[is_today_valid]
    is_today_invalid = mask_night2 & (~s_date.isin(valid_trading_days_set))
    valid_n2_mask = is_today_invalid & s_next_trade_day.notna()
    trade_dates.loc[valid_n2_mask] = s_next_trade_day.loc[valid_n2_mask]

    df["_trade_date"] = trade_dates
    df = df.dropna(subset=["_trade_date"])
    df = df[df["_trade_date"].isin(valid_trading_days_set)]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "instrument_id",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        )

    df = df.sort_values(["instrument_id", "_trade_date", "datetime"])

    # 4, 重采样
    # =================================================================
    # [性能优化2]成交量增量计算向量化 (替代groupby().apply())
    # =================================================================
    # 第一步: 计算原始差值
    df["vol_diff"] = df.groupby(["instrument_id", "_trade_date"])["volume"].diff()
    # 第二步: 标记复位 (成交量减少视为复位)
    df["is_reset"] = df["vol_diff"] < 0
    # 第三步: 生成分组ID (遇到复位就累加, 形成新的子组)
    df["reset_group"] = df.groupby(["instrument_id", "_trade_date"])[
        "is_reset"
    ].cumsum()  # 这一步巧妙地用cumsum替代了复杂的逻辑判断
    # 第四步: 在子组内计算增量
    df["vol_increment"] = (
        df.groupby(["instrument_id", "_trade_date", "reset_group"])["volume"]
        .diff()
        .fillna(0)
    )
    # 确保非负
    df["vol_increment"] = df["vol_increment"].clip(lower=0)
    # 清理临时列 (可选, 节省内存)
    df.drop(["vol_diff", "is_reset", "reset_group"], axis=1, inplace=True)

    results = []
    groups = df.groupby(["instrument_id", "_trade_date"], sort=False, group_keys=False)
    # 处理日级K线
    if time == "1D":
        # 直接按交易日聚合, 生成日级K线
        for (inst_id, t_date), group in groups:
            if group.empty:
                continue

            # 计算日级K线指标
            open_price = group["last_price"].iloc[0]
            high_price = group["last_price"].max()
            low_price = group["last_price"].min()
            close_price = group["last_price"].iloc[-1]
            total_volume = group["vol_increment"].sum()
            open_interest = group["open_interest"].iloc[-1]  # 取最后的持仓量

            # 构建日级K线记录
            daily_kline = pd.DataFrame(
                {
                    "trading_day": [pd.Timestamp(t_date)],
                    "instrument_id": [inst_id],
                    "open": [open_price],
                    "high": [high_price],
                    "low": [low_price],
                    "close": [close_price],
                    "volume": [total_volume],
                    "open_interest": [open_interest],
                }
            )

            results.append(daily_kline)

        if not results:
            return pd.DataFrame(
                columns=[
                    "trading_day",
                    "instrument_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                ]
            )

        final_df = pd.concat(results, ignore_index=True)

        return final_df.sort_values(["instrument_id", "trading_day"]).reset_index(
            drop=True
        )

    # 处理分钟/小时级别K线
    else:
        for (inst_id, t_date), group in groups:
            if group.empty:
                continue

            g = group[
                ["datetime", "last_price", "vol_increment", "open_interest"]
            ].copy()

            try:
                r = g.resample(time, on="datetime", closed="left", label="left").agg(
                    {
                        "last_price": ["first", "max", "min", "last"],
                        "vol_increment": "sum",
                        "open_interest": "last",
                    }
                )
                # 展开多级列名
                r.columns = ["_".join(col).strip() for col in r.columns]
                # 重命名为目标列名
                r = r.rename(
                    columns={
                        "last_price_first": "open",
                        "last_price_max": "high",
                        "last_price_min": "low",
                        "last_price_last": "close",
                        "vol_increment_sum": "volume",
                        "open_interest_last": "open_interest",
                    }
                )

                # r = trigger_based_resample(
                #     g, time_col="datetime", count_limit=int(time[:-3])
                # ).agg(
                #     {
                #         "datetime": "first",
                #         "last_price": ["first", "max", "min", "last"],
                #         "vol_increment": "sum",
                #         "open_interest": "last",
                #     }
                # )
                # # 展开多级列名
                # r.columns = ["_".join(col).strip() for col in r.columns]
                # # 重命名为目标列名
                # r = r.rename(
                #     columns={
                #         "datetime_first": "datetime",
                #         "last_price_first": "open",
                #         "last_price_max": "high",
                #         "last_price_min": "low",
                #         "last_price_last": "close",
                #         "vol_increment_sum": "volume",
                #         "open_interest_last": "open_interest",
                #     }
                # )
                # r.set_index("datetime", inplace=True)

                r = r.dropna(subset=["open"])
                if not r.empty:
                    r["instrument_id"] = inst_id
                    r["trade_date"] = t_date
                    results.append(r.reset_index())
            except Exception as e:
                print(f"Error processing {inst_id} on {t_date}: {e}")
                continue

        if not results:
            return pd.DataFrame(
                columns=[
                    "datetime",
                    "instrument_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "open_interest",
                ]
            )

        final_df = pd.concat(results, ignore_index=True)
        target_cols = [
            "datetime",
            "instrument_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "trade_date",
        ]
        final_cols = [c for c in target_cols if c in final_df.columns]

        return (
            final_df[final_cols]
            .sort_values(["instrument_id", "datetime"])
            .reset_index(drop=True)
        )


def cumulative_cal_return(df, calc_type="prod"):
    """
    功能: 根据交易信号计算盈亏 (为了简化代码逻辑提高运算效率, 一次调用只能计算一个方向上的盈亏) , 支持收益率版本('prod')和点差版本('sum')

    参数:
        df(pd.DataFrame): 有datetime, last_price和trade_signal等列
        calc_type(str): 计算类型, 'prod'表示收益率版本, 'sum'表示点差版本

    返回:
        list
    """
    # 数据预处理
    df = df.sort_values(by=["datetime"]).reset_index(drop=True)
    df = df.dropna(subset=["last_price", "trade_signal"])

    if df.empty:
        return []

    # 计算收益率或点差
    if calc_type == "prod":
        df["return"] = df["last_price"].pct_change().fillna(0) + 1
        # 第一个值设为1 (无变化)
        df.loc[0, "return"] = 1
    elif calc_type == "sum":
        df["return"] = df["last_price"].diff().fillna(0)
        # 第一个值设为0 (无变化)
        df.loc[0, "return"] = 0
    else:
        raise ValueError("calc_type must be either 'prod' or 'sum'")

    # 获取交易信号
    signals = df["trade_signal"].values
    returns = df["return"].values

    # 找到所有开仓信号(1)和平仓信号(0)的位置
    open_positions = np.where(signals == 1)[0]
    close_positions = np.where(signals == 0)[0]

    results = []

    # 配对开仓和平仓信号
    open_idx = 0
    close_idx = 0

    while open_idx < len(open_positions) and close_idx < len(close_positions):
        open_pos = open_positions[open_idx]
        close_pos = close_positions[close_idx]

        # 确保平仓时间在开仓之后
        if close_pos <= open_pos:
            close_idx += 1
            continue

        # 计算从开仓到平仓期间的收益
        start_pos = open_pos + 1  # 不包含开仓当天的数据
        end_pos = close_pos + 1  # 包含平仓当天的数据

        if calc_type == "prod":
            # 收益率版本: 计算乘积后减1
            period_returns = returns[start_pos:end_pos]
            total_return = np.prod(period_returns) - 1
        else:  # calc_type == 'sum'
            # 点差版本: 计算总和
            period_returns = returns[start_pos:end_pos]
            total_return = np.sum(period_returns)

        results.append(total_return)

        # 移动到下一组信号
        open_idx += 1
        close_idx += 1

    return results


def filter_trade_signals(df):
    """
    功能: 过滤向量化回测生成的无用交易信号 (为了简化代码逻辑提高运算效率, 一次调用只能过滤一个方向上的信号)

    参数:
        df(pd.DataFrame): 有datetime trade_signal等列

    返回:
        df(pd.DataFrame): 过滤后的交易信号
    """
    if df.empty or "trade_signal" not in df.columns:
        return df

    trade_signals = df["trade_signal"].to_numpy()

    # 找出所有值为1和0的下标
    ones_indices = np.where(trade_signals == 1)[0]
    zeros_indices = np.where(trade_signals == 0)[0]

    if len(ones_indices) == 0 or len(zeros_indices) == 0:
        # 如果没有1或0, 直接返回原DataFrame (trade_signal列全为NaN)
        result = df.copy()
        result["trade_signal"] = np.nan
        return result

    # 初始化结果数组
    filtered_signals = np.full(len(trade_signals), np.nan)

    # 使用更清晰的双指针逻辑
    one_idx, zero_idx = 0, 0
    last_valid_end = -1
    while one_idx < len(ones_indices) and zero_idx < len(zeros_indices):
        # 寻找下一个有效的1 (下标必须大于last_valid_end)
        while one_idx < len(ones_indices) and ones_indices[one_idx] <= last_valid_end:
            one_idx += 1

        if one_idx >= len(ones_indices):
            break

        current_one_pos = ones_indices[one_idx]

        # 寻找下一个有效的0 (下标必须大于当前1的位置)
        while (
            zero_idx < len(zeros_indices) and zeros_indices[zero_idx] <= current_one_pos
        ):
            zero_idx += 1

        if zero_idx >= len(zeros_indices):
            break

        current_zero_pos = zeros_indices[zero_idx]

        # 标记这对信号有效
        filtered_signals[current_one_pos] = 1
        filtered_signals[current_zero_pos] = 0

        # 更新last_valid_end为当前0的位置
        last_valid_end = current_zero_pos

        # 移动指针
        one_idx += 1
        zero_idx += 1

    # 创建结果DataFrame
    result_df = df.copy()
    result_df["trade_signal"] = filtered_signals

    return result_df


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


# 2, 遍历回测所用类或函数
# 动态生成K线数据
class K_line:
    def __init__(self, open_time, price):
        self.open_time = open_time
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = 0.0  # 这里储存的是累计成交量

    def update(self, price, delta_volume):
        self.close = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.volume += delta_volume

    def finalize(self):
        return {
            "open_time": self._format_time(self.open_time),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @staticmethod
    def _format_time(timestamp):
        return pd.Timestamp(timestamp, unit="s")

    def __str__(self):
        return f"[{self._format_time(self.open_time)}] O:{self.open} H:{self.high} L:{self.low} C:{self.close} V:{self.volume}"


class K_line_generator:
    def __init__(self, interval_minutes=1):
        self.interval_seconds = interval_minutes * 60
        self.current_kline = None
        self.klines = []
        self.last_volume = 0.0  # 储存上一个Tick的累计成交量
        self.last_price = None  # 可选: 用于判断是否为第一个Tick

    def add_tick(self, tick_time, price, cumulative_volume):
        """
        功能: 添加一条Tick数据

        参数:
            tick_time: 时间戳 (秒)
            price: 成交价
            cumulative_volume: 累计成交量
        """
        # 计算新增成交量
        if self.last_price is None:
            delta_volume = 0.0  # 第一个Tick, 无法计算增量
        else:
            delta_volume = cumulative_volume - self.last_volume

        # 对齐K线周期
        kline_open_time = (tick_time // self.interval_seconds) * self.interval_seconds

        if self.current_kline is None:
            # 初始化第一根K线
            self.current_kline = K_line(kline_open_time, price)
            self.current_kline.update(price, delta_volume)
        elif kline_open_time != self.current_kline.open_time:
            # 当前K线结束, 保存并新建下一根
            self.klines.append(self.current_kline.finalize())
            self.current_kline = K_line(kline_open_time, price)
            self.current_kline.update(price, delta_volume)
        else:
            # 更新当前K线
            self.current_kline.update(price, delta_volume)

        # 更新当前K线
        self.last_price = price
        self.last_volume = cumulative_volume

    def flush(self):
        """
        功能: 输出当前正在构建的K线 (如果有的话)
        """
        if self.current_kline:
            self.klines.append(self.current_kline.finalize())
            self.current_kline = None

    def get_klines(self):
        return self.klines

    def print_klines(self):
        for k in self.get_klines():
            print(k)


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
    df3_1 = tick_to_K(df2_2, "15min")
    df3_1 = df3_1.reset_index()

    df3_1["ma5"] = MA(df3_1["close"], 5)
    df3_1["ma10"] = MA(df3_1["close"], 10)
    df3_1["ma20"] = MA(df3_1["close"], 20)
    df3_1["ma40"] = MA(df3_1["close"], 40)
    df3_1["ma60"] = MA(df3_1["close"], 60)

    df3_1["local_min"] = (
        df3_1["low"]
        .rolling(window=13)
        .apply(find_local_extrema, raw=False, args=(13, 9, "low"))
    )
    df3_1.loc[df3_1["local_min"] == False, "local_min"] = np.nan
    df3_1.loc[df3_1["local_min"] == True, "local_min"] = df3_1.loc[
        df3_1["local_min"] == True, "low"
    ]
    df3_1["local_max"] = (
        df3_1["high"]
        .rolling(window=13)
        .apply(find_local_extrema, raw=False, args=(13, 9, "high"))
    )
    df3_1.loc[df3_1["local_max"] == False, "local_max"] = np.nan
    df3_1.loc[df3_1["local_max"] == True, "local_max"] = df3_1.loc[
        df3_1["local_max"] == True, "high"
    ]

    df3_1["close_open"] = df3_1["close"] / df3_1["open"] - 1
    df3_1["close_open_0.05"] = (
        df3_1["close_open"].expanding().apply(find_per_value, raw=False, args=(0.05,))
    )
    df3_1["close_open_0.95"] = (
        df3_1["close_open"].expanding().apply(find_per_value, raw=False, args=(0.95,))
    )
    df3_1["volume_0.95_percentile"] = (
        df3_1["volume"].expanding().apply(find_per_value, raw=False, args=(0.95,))
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
        df3_1["rsi"].expanding().apply(find_per_value, raw=False, args=(0.05,))
    )
    df3_1["rsi_0.95_percentile"] = (
        df3_1["rsi"].expanding().apply(find_per_value, raw=False, args=(0.95,))
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

    df3_2 = tick_to_K(df2_2, "1D")
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
