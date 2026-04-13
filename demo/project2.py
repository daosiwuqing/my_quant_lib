# 该项目的作用: 观察策略的进出场信号表现
import pandas as pd
import numpy as np
import bisect

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.animation import FuncAnimation

# 设置全局字体
fm.fontManager.addfont(
    "C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)
mpl.rcParams["font.sans-serif"] = "FZHei-B01S"
mpl.rcParams["axes.unicode_minus"] = False
font = fm.FontProperties(
    fname="C:\\Users\\29433\\AppData\\Local\\Microsoft\\Windows\\Fonts\\FZHTJW.TTF"
)

import sys

sys.path.append("D:\\LearningAndWorking\\VSCode\\python\\module")
from MyTT import *


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


class K_line_analyzer:
    def __init__(self):
        self.custom_style = self._create_custom_style()

    def _create_custom_style(self):
        try:
            market_colors = mpf.make_marketcolors(
                up="red",
                down="green",
                wick={"up": "red", "down": "green"},
                edge={"up": "red", "down": "green"},
                volume={"up": "red", "down": "green"},
            )

            style_kwargs = {
                "base_mpf_style": "classic",
                "marketcolors": market_colors,
                "gridcolor": "gray",
                "gridstyle": "--",
                "facecolor": "white",
                "figcolor": "white",
                "edgecolor": "white",
                # 强制图表内部使用中文字体
                "rc": {
                    # 1. 强制设置字体族
                    "font.family": "sans-serif",
                    # 2. 将中文字体放在列表首位, 确保优先匹配, 同时也加上了常见的备选字体
                    "font.sans-serif": [
                        "FZHei-B01S",
                        "SimHei",
                        "Heiti TC",
                        "Arial Unicode MS",
                        "DejaVu Sans",
                    ],
                    # 3. 解决负号乱码
                    "axes.unicode_minus": False,
                    # 4. (可选) 调整分辨率
                    "figure.dpi": 100,
                },
            }
            return mpf.make_mpf_style(**style_kwargs)

        except Exception as e:
            print(f"警告: 自定义样式创建失败, 原因: {e}")
            print("正在使用备用极简样式...")
            # 备用样式同样必须包含 rc, 否则会再次报错
            return mpf.make_mpf_style(
                marketcolors=mpf.make_marketcolors(up="r", down="g"),
                rc={
                    "font.family": "sans-serif",
                    "font.sans-serif": ["FZHei-B01S", "SimHei"],
                    "axes.unicode_minus": False,
                },
            )

    def _validate_dataframe(self, df, required_columns):
        """
        功能: 验证df格式, 检查是否有日期时间列, OHLC列和信号列
        """
        if df is None or df.empty:
            return False

        # 检查是否有日期时间列, OHLC列和信号列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"df缺少核心列: {missing_cols}")
        return True

    def _extract_by_count(self, df, target_time, count_before=10, count_after=10):
        """
        功能: 向前查找最接近的时间点并截取数据
        """
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        matched_time = None
        center_idx = -1

        # 使用merge_asof进行向前匹配
        target_series = pd.DataFrame({"target_time": [target_time]})
        index_series = pd.DataFrame({"df_time": df.index})

        try:
            merged = pd.merge_asof(
                target_series,
                index_series,
                left_on="target_time",
                right_on="df_time",
                direction="backward",
            )

            if not merged["df_time"].isna().any():
                matched_time = merged["df_time"].iloc[0]
                center_idx = df.index.get_loc(matched_time)
            else:
                center_idx = 0
                matched_time = df.index[0]
        except:
            # 回退逻辑
            past_indices = df.index[df.index <= target_time]
            if len(past_indices) > 0:
                matched_time = past_indices[-1]
                center_idx = df.index.get_loc(matched_time)
            else:
                center_idx = 0
                matched_time = df.index[0]

        start_idx = max(0, center_idx - count_before)
        end_idx = min(len(df) - 1, center_idx + count_after)

        return df.iloc[start_idx : end_idx + 1], matched_time

    def _prepare_add_plots(self, df, panel_config):
        """
        功能: 根据特定配置准备附加图表

        参数:
            df: 数据
            panel_config: 该频率特有的配置字典 (例如 {'volume': {...}, 'ma': {...}})
        """
        add_plots = []
        if not panel_config:
            return add_plots

        for col_name, config in panel_config.items():
            if col_name in df.columns:
                plot_config = {"data": df[col_name], **config}

                # 默认面板为0 (主图) , 除非指定
                panel = plot_config.get("panel", 0)

                if config["type"] == "bar":
                    add_plots.append(
                        mpf.make_addplot(
                            plot_config["data"],
                            panel=panel,
                            type="bar",
                            color=plot_config.get("color", "lightgray"),
                            label=col_name,
                        )
                    )
                elif config["type"] == "line":
                    add_plots.append(
                        mpf.make_addplot(
                            plot_config["data"],
                            panel=panel,
                            color=plot_config.get("color", "blue"),
                            linewidths=plot_config.get("linewidths", 1),
                            label=col_name,
                        )
                    )
        return add_plots

    def _create_plots(self, dfs, titles, additional_plots_configs=None):
        """
        功能: 在内存中同时绘制多张独立的K线图, 最后统一弹窗显示
        """
        # 1. 配置处理
        if additional_plots_configs is None:
            additional_plots_configs = [None] * len(dfs)
        while len(additional_plots_configs) < len(dfs):
            additional_plots_configs.append(None)

        # 2. 准备一个列表来存放生成的 “画布”
        all_figures = []

        print(f"正在绘制 {len(dfs)} 张图表...")

        # 3. 循环绘制 (此时只是在内存中画, 不会弹窗)
        for i, (df, title) in enumerate(zip(dfs, titles)):
            config = additional_plots_configs[i]
            add_plots = self._prepare_add_plots(df, config)

            # 使用returnfig=True, 这行代码不会显示窗口, 而是把画布对象 (fig) 返回给我们, 我们可以把它存起来, 等所有图都画完了再统一显示
            if add_plots:
                fig, axlist = mpf.plot(
                    df,
                    type="candle",
                    style=self.custom_style,
                    figsize=(15, 8),
                    title=title,
                    addplot=add_plots,
                    returnfig=True,
                    warn_too_much_data=5000,
                )
            else:
                fig, axlist = mpf.plot(
                    df,
                    type="candle",
                    style=self.custom_style,
                    figsize=(15, 8),
                    title=title,
                    returnfig=True,
                    warn_too_much_data=5000,
                )

            # 把画好的画布存起来
            all_figures.append(fig)

        # 4. 所有图画完后, 统一显示, 这一步会触发GUI事件, 把所有存好的图一次性弹出来
        plt.show()

        print("绘制完成！")
        return all_figures

    def analyze_trade_signals(
        self,
        dfs,
        instrument_id1,
        counts_before=None,
        counts_after=None,
        signal_column="trade_signal",
        additional_plots_configs=None,
        frequency_names=None,
    ):
        """
        参数:
            additional_plots_configs(list of dicts): 每个dict对应一个频率的附加绘图配置, 顺序必须与dfs一致.
                示例:
                [
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
        """
        if not dfs or len(dfs) == 0:
            print("错误: 至少需要一个数据框")
            return

        # 核心列只检查日期时间列, OHLC和信号列
        required_columns = ["datetime", "open", "high", "low", "close", signal_column]

        processed_dfs = []
        for df in dfs:
            self._validate_dataframe(df, required_columns)
            df_copy = df.set_index("datetime").copy()
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)
            df_copy = df_copy.sort_index()
            processed_dfs.append(df_copy)

        n_freq = len(dfs)

        if counts_before is None:
            counts_before = [10] * n_freq
        if counts_after is None:
            counts_after = [10] * n_freq

        if frequency_names is None:
            default_names = ["High Freq", "Mid Freq", "Low Freq"]
            frequency_names = default_names[:n_freq]

        high_freq_df = processed_dfs[0]
        signal_indices = high_freq_df[high_freq_df[signal_column] == 1].index

        if len(signal_indices) == 0:
            print(f"警告: 高频数据中无信号")
            return

        print(f"发现 {len(signal_indices)} 个高频信号, 开始分析...")

        for i, high_freq_time in enumerate(signal_indices):
            print(f"\n--- [{i+1}/{len(signal_indices)}] 高频时间: {high_freq_time} ---")

            valid_dfs = []
            valid_titles = []

            # --- A. 处理高频 ---
            window_df_hf, matched_time_hf = self._extract_by_count(
                high_freq_df, high_freq_time, counts_before[0], counts_after[0]
            )
            valid_dfs.append(window_df_hf)
            valid_titles.append(f"{instrument_id1} {frequency_names[0]}")

            # --- B. 处理中/低频 ---
            for j in range(1, n_freq):
                lower_freq_df = processed_dfs[j]

                window_df_lf, matched_time_lf = self._extract_by_count(
                    lower_freq_df, high_freq_time, counts_before[j], counts_after[j]
                )

                # 检查信号
                has_signal = False
                if matched_time_lf in lower_freq_df.index:
                    signal_val = lower_freq_df.loc[matched_time_lf, signal_column]
                    if signal_val == 1:
                        has_signal = True

                if has_signal:
                    print(
                        f"  [√] {frequency_names[j]} 匹配: {matched_time_lf} (有信号)"
                    )
                    valid_dfs.append(window_df_lf)
                    valid_titles.append(f"{instrument_id1} {frequency_names[j]}")
                else:
                    print(
                        f"  [x] {frequency_names[j]} 匹配: {matched_time_lf} (无信号)"
                    )

            # --- C. 绘图 ---
            if len(valid_dfs) == len(dfs):
                self._create_plots(valid_dfs, valid_titles, additional_plots_configs)
                plt.show(block=True)
                plt.close("all")

        print("分析完成")


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
    df3_1 = tick_to_K(df2_2, "5min")

    df3_1["rsi"] = RSI(df3_1["close"], 20)
    df3_1["rsi_0.05_percentile"] = (
        df3_1["rsi"].expanding().apply(find_per_value, raw=False, args=(0.05,))
    )
    df3_1["rsi_0.95_percentile"] = (
        df3_1["rsi"].expanding().apply(find_per_value, raw=False, args=(0.95,))
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
    df3_2 = tick_to_K(df2_2, "1h")

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
    df3_3 = tick_to_K(df2_2, "1D")

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
    plot = K_line_analyzer()
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
