# 该脚本的作用: 封装常用的函数


# 该类的作用: 封装在指数回测计算中常用的函数
import pandas as pd
import re, bisect


class Cal_index1(object):
    editor = "zhaobo"

    def __init__(self):
        self.log = "2026/03/31对包进行了修改"

    def get_maturity_date(instrument_id, now_date, maturity_day=28, max_years_ahead=3):
        """
        功能: 从instrument_id中提取出合约到期日

        参数:
            instrument_id(str): 合约代码, 例如'IF2309'
            now_date(pd.Timestamp): 交易日
            maturity_day(int): 到期日日期 (默认28)
            max_years_ahead(int): 允许的最大未来年限 (默认3年)

        返回:
            pd.Timestamp: 合约到期日
        """
        # 1, 提取合约代码中的年份个位和月份
        match = re.search(r"(\d{3})$", instrument_id)
        if not match:
            raise ValueError(f"合约代码{instrument_id}格式错误, 未找到末尾3位数字")

        num_part = match.group(1)
        y_digit = int(num_part[-3])  # 年份个位
        month = int(num_part[-2:])  # 月份

        if not (1 <= month <= 12):
            raise ValueError(f"合约代码{instrument_id}中月份{month}无效")

        # 确保输入是Timestamp
        if not isinstance(now_date, pd.Timestamp):
            raise TypeError("now_date必须是pd.Timestamp类型")

        current_date = now_date
        current_year = current_date.year

        # 2, 在有限的时间窗口内寻找最佳年份
        for offset in range(max_years_ahead + 1):
            candidate_year = current_year + offset

            # 核心匹配逻辑: 年份个位必须一致
            if candidate_year % 10 == y_digit:
                try:
                    # 构造候选到期日
                    cand_date = pd.Timestamp(
                        year=candidate_year, month=month, day=maturity_day
                    )
                except Exception as e:
                    # 如果日期非法 (如2月30日) , 跳过
                    continue

                # 必须是未来或当天
                if cand_date >= current_date:
                    return cand_date

        # 如果循环结束都没返回, 说明在指定年限内没有有效合约
        raise ValueError(
            f"合约{instrument_id}无有效到期日: 在{max_years_ahead}年内未找到匹配的未来日期"
        )

    def cal_date_spread(current_date, maturity_date, trading_calender):
        """
        功能: 计算当前日期距离合约到期日还有几个月, 并计算当月已经过了多少个交易日以及还剩多少个交易日

        参数:
            current_date(pd.Timestamp): 当前交易日
            maturity_date(pd.Timestamp): 合约到期日
            trading_calender(list): 已按升序排列的交易日列表 (包含pd.Timestamp对象)

        返回:
            dict: 包含剩余月数, 当月已过/剩余交易日数等信息
        """

        # 基础校验
        if trading_calender.empty:
            raise ValueError("交易日序列不能为空")

        # 确保输入是pd.Timestamp
        if not isinstance(current_date, pd.Timestamp):
            raise TypeError("current_date必须是pd.Timestamp类型")
        if not isinstance(maturity_date, pd.Timestamp):
            raise TypeError("maturity_date必须是pd.Timestamp类型")

        # 校验列表中元素类型 (可选, 为了健壮性)
        if not isinstance(trading_calender[0], pd.Timestamp):
            raise TypeError("trading_calendar列表中的元素必须是pd.Timestamp类型")

        # 使用bisect_left进行二分查找, 定位当前日期在列表中的索引
        idx = bisect.bisect_left(trading_calender, current_date)

        # 校验找到的索引是否有效, 且确实匹配当前日期
        if idx >= len(trading_calender) or trading_calender[idx] != current_date:
            raise ValueError(f"当前日期{current_date}不在交易日序列中, 或序列已结束")

        current_year = current_date.year
        current_month = current_date.month

        # 1, 计算剩余月数
        # 逻辑: (目标年 - 当前年) * 12 + (目标月 - 当前月), 忽略具体日期
        months_remaining = (maturity_date.year - current_year) * 12 + (
            maturity_date.month - current_month
        )
        if months_remaining < 0:
            months_remaining = 0

        # 2, 双向遍历统计当月交易日
        days_passed = 0
        days_left = 0
        # 2.1, 向前统计(已过交易日, 含当天), 从idx开始倒序遍历, 直到月份改变或列表开头
        for i in range(idx, -1, -1):
            t_date = trading_calender[i]
            # pd.Timestamp支持直接访问.year和.month
            if t_date.year == current_year and t_date.month == current_month:
                days_passed += 1
            else:
                break

        # 2.2, 向后统计(剩余交易日, 含当天), 从idx开始正序遍历, 直到月份改变或列表结尾
        for i in range(idx, len(trading_calender)):
            t_date = trading_calender[i]
            if t_date.year == current_year and t_date.month == current_month:
                days_left += 1
            else:
                break

        # 构造返回字典
        return {
            "current_date": current_date,
            "maturity_date": maturity_date,
            "months_remaining": months_remaining,
            "trading_days_passed_this_month": days_passed,
            "trading_days_left_this_month": days_left,
            "total_trading_days_this_month": days_passed + days_left - 1,
        }

    def add_contract_info(codes_series):
        """
        功能: 为入选品种(code)添加合约规模(scale), 保证金比例(margin), 手续费(fee)等合约基本数据

        参数:
            codes_series(pd.Series): 品种代码

        返回:
            df(pd.DataFrame): 含有scale, margin等列, code作为索引
        """
        info_dict = {
            "A": [10, 0.08],
            "AD": [10, 0.1],
            "AG": [15, 0.09],
            "AL": [5, 0.08],
            "AO": [20, 0.09],
            "AP": [10, 0.1],
            "AU": [1000, 0.08],
            "B": [10, 0.08],
            "BB": [500, 0.4],
            "BC": [5, 0.08],
            "BR": [5, 0.12],
            "BU": [10, 0.1],
            "BZ": [30, 0.08],
            "C": [10, 0.08],
            "CF": [5, 0.07],
            "CJ": [5, 0.12],
            "CS": [10, 0.06],
            "CU": [5, 0.08],
            "CY": [5, 0.07],
            "EB": [5, 0.08],
            "EG": [10, 0.08],
            "FB": [10, 0.1],
            "FG": [20, 0.09],
            "FU": [10, 0.1],
            "HC": [10, 0.07],
            "I": [100, 0.13],
            "IC": [200, 0.08],
            "IF": [300, 0.08],
            "IH": [300, 0.08],
            "IM": [200, 0.08],
            "J": [100, 0.2],
            "JD": [5, 0.08],
            "JM": [60, 0.2],
            "JR": [20, 0.15],
            "L": [5, 0.07],
            "LC": [1, 0.09],
            "LG": [90, 0.08],
            "LH": [16, 0.12],
            "LR": [20, 0.15],
            "LU": [10, 0.1],
            "M": [10, 0.07],
            "MA": [10, 0.08],
            "NI": [1, 0.12],
            "NR": [10, 0.08],
            "OI": [10, 0.09],
            "OP": [40, 0.09],
            "P": [10, 0.08],
            "PB": [5, 0.08],
            "PD": [1000, 0.19],
            "PF": [5, 0.08],
            "PG": [20, 0.08],
            "PK": [5, 0.08],
            "PL": [20, 0.07],
            "PM": [50, 0.15],
            "PP": [5, 0.07],
            "PR": [15, 0.07],
            "PS": [3, 0.15],
            "PT": [1000, 0.19],
            "PX": [5, 0.07],
            "RB": [10, 0.07],
            "RI": [20, 0.15],
            "RM": [10, 0.09],
            "RR": [10, 0.06],
            "RS": [10, 0.2],
            "RU": [10, 0.08],
            "SA": [20, 0.09],
            "SC": [1000, 0.1],
            "SF": [5, 0.12],
            "SH": [30, 0.08],
            "SI": [5, 0.09],
            "SM": [5, 0.12],
            "SN": [1, 0.12],
            "SP": [10, 0.08],
            "SR": [10, 0.07],
            "SS": [5, 0.07],
            "T": [10000, 0.02],
            "TA": [5, 0.07],
            "TF": [10000, 0.01],
            "TL": [10000, 0.035],
            "TS": [20000, 0.005],
            "UR": [10, 0.08],
            "V": [5, 0.07],
            "WH": [20, 0.15],
            "WR": [10, 0.09],
            "Y": [10, 0.07],
            "ZC": [100, 0.05],
            "ZN": [5, 0.08],
        }
        codes = codes_series.index
        info_values = [info_dict.get(code, [None, None]) for code in codes]
        scale = [val[0] for val in info_values]
        margin = [val[1] for val in info_values]
        df_info = pd.DataFrame({"scale": scale, "margin": margin}, index=codes)
        df_original = codes_series.to_frame()
        result_df = pd.concat([df_original, df_info], axis=1)

        return result_df

    def get_weight(index, year_month):
        """
        功能：从'南华指数系列历史权重.xlsx'中获取某个指数某个日期的权重数据

        参数:
            index(str): 工作表名称，也是指数的简称，如"综合指数"
            year_month(str): 权重调整日期，如"2020-06"

        参数:
            weight(pd.Series): 权重数据
        """
        df_raw = pd.read_excel(
            "D:\\LearningAndWorking\\VSCode\\data\\xlsx\\南华指数系列历史权重(2025).xlsx",
            sheet_name=index,
        ).iloc[:-1, :]
        regex_pattern = f"{year_month}|代码"
        df_filtered = df_raw.filter(regex=regex_pattern, axis=1)
        df_filtered.set_index("代码", inplace=True)
        weight = df_filtered.squeeze()

        return weight[(weight != 0) & (weight.notna())]


# 该类的作用：封装在期权计算中常用的函数
import pandas as pd
import numpy as np
import numba
import matplotlib.pyplot as plt
import scipy as sp


class Pri_option(object):
    editor = "zhaobo"

    def __init__(self):
        self.log1 = "2024/01/23对包进行了修改"

    # 1、布朗运动和伊藤引理
    # 1.1、标准布朗运动
    def standard_brownian(steps, paths, T, S0):
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵，用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = S_path[step - 1] + rn[step - 1] * np.sqrt(dt)
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = standard_brownian(steps=100, paths=10, T=1, S0=0)

    # 1.2、广义的布朗运动
    def brownian(steps, paths, T, S0, a, b):
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵，用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = (
                S_path[step - 1] + a * dt + b * rn[step - 1] * np.sqrt(dt)
            )  # 和标准布朗运动的区别就在这一行
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = brownian(steps=100, paths=10, T=1, S0=0, a=5, b=2)

    # 1.3、几何布朗运动
    def geo_brownian(steps, paths, T, S0, u, sigma):
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵，用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = S_path[step - 1] * np.exp(
                (u - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rn[step]
            )  # 和其他布朗运动的区别就在这一行
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = geo_brownian(steps=100, paths=50, T=1, S0=100, u=0.03, sigma=0.2)

    # 2、BSM公式
    # 2.1、定价公式
    def BSM(CP, S, X, sigma, T, r, b):
        """
        Parameters
        ----------
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        sigma : 波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b=r时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        Returns
        ----------
        返回欧式期权的估值
        """
        d1 = (np.log(S / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if CP == "C":
            value = S * np.exp((b - r) * T) * sp.stats.norm.cdf(d1) - X * np.exp(
                -r * T
            ) * sp.stats.norm.cdf(d2)
        else:
            value = X * np.exp(-r * T) * sp.stats.norm.cdf(-d2) - S * np.exp(
                (b - r) * T
            ) * sp.stats.norm.cdf(-d1)

        return value

    # BSM(CP="C", S=100, X=95, sigma=0.25, T=1, r=0.03, b=0.03)

    # 2.2、二分法求解隐含波动率
    def binary(V0, CP, S, X, T, r, b, vol_est=0.2):
        """
        Parameters
        ----------
        V0 : 期权价值.
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b=r时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        vol_est : 预计的初始波动率.
        Returns
        ----------
        返回看涨期权的隐含波动率。
        """
        start = 0  # 初始波动率下限
        end = 2  # 初始波动率上限
        e = 1  # 先给定一个值，让循环运转起来
        while abs(e) >= 0.0001:  # 迭代差异的精度，根据需要调整
            try:
                val = Pri_option.BSM(CP, S, X, vol_est, T, r, b)
            except ZeroDivisionError:
                print("期权的内在价值大于期权的价格，无法收敛出波动率，会触发除0错误！")
                break
            if val - V0 > 0:  # 若计算的期权价值大于实际价值，说明使用的波动率偏大
                end = vol_est
                vol_est = (start + end) / 2
                e = end - vol_est
            else:  # 若计算的期权价值小于实际价值，说明使用的波动率偏小
                start = vol_est
                vol_est = (start + end) / 2
                e = start - vol_est

        return round(vol_est, 4)

    # value = BSM(CP="C", S=100, X=95, T=1, sigma=0.25, r=0.03, b=0.03) #实验一个期权的价值
    # print(value)
    # vol = binary(V0=value, CP="C", S=100, X=95, T=1, r=0.03, b=0.03)  #根据刚才实验的期权价值求一下波动率是否正确
    # print(vol)

    # 2.3 牛顿法求解隐含波动率
    def newton(V0, CP, S, X, T, r, b, vol_est=0.25, n_iter=1000):
        # n_iter表示迭代的次数
        for i in range(n_iter):
            d1 = (np.log(S / X) + (b + vol_est**2 / 2) * T) / (vol_est * np.sqrt(T))
            vega = S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5  # 计算vega
            vol_est = (
                vol_est - (Pri_option.BSM(CP, S, X, vol_est, T, r, b) - V0) / vega
            )  # 每次迭代都重新算一下波动率

        return vol_est

    # vol = newton(V0, CP, S, X, T, r, b, vol_est=0.2, n_iter=1000)
    # print(vol)

    # 3、欧式期权希腊字母计算
    # 3.1、解析解下的实现
    def greeks(CP, S, X, sigma, T, r, b):  # 计算greeks的函数
        """
        Parameters
        ----------
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        sigma : 波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b=r时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        Returns
        ----------
        返回欧式期权的估值和希腊字母
        """
        d1 = (np.log(S / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))  # 求d1
        d2 = d1 - sigma * np.sqrt(T)  # 求d2

        if CP == "C":
            option_value = S * np.exp((b - r) * T) * sp.stats.norm.cdf(d1) - X * np.exp(
                -r * T
            ) * sp.stats.norm.cdf(
                d2
            )  # 计算期权价值
            delta = np.exp((b - r) * T) * sp.stats.norm.cdf(d1)
            gamma = (
                np.exp((b - r) * T) * sp.stats.norm.pdf(d1) / (S * sigma * T**0.5)
            )  # 注意是pdf，概率密度函数
            vega = S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5  # 计算vega
            theta = (
                -np.exp((b - r) * T) * S * sp.stats.norm.pdf(d1) * sigma / (2 * T**0.5)
                - r * X * np.exp(-r * T) * sp.stats.norm.cdf(d2)
                - (b - r) * S * np.exp((b - r) * T) * sp.stats.norm.cdf(d1)
            )
            if b != 0:  # rho比较特别，b是否为0会影响求导结果的形式
                rho = X * T * np.exp(-r * T) * sp.stats.norm.cdf(d2)
            else:
                rho = -np.exp(-r * T) * (
                    S * sp.stats.norm.cdf(d1) - X * sp.stats.norm.cdf(d2)
                )

        else:
            option_value = X * np.exp(-r * T) * sp.stats.norm.cdf(-d2) - S * np.exp(
                (b - r) * T
            ) * sp.stats.norm.cdf(-d1)
            delta = -np.exp((b - r) * T) * sp.stats.norm.cdf(-d1)
            gamma = (
                np.exp((b - r) * T) * sp.stats.norm.pdf(d1) / (S * sigma * T**0.5)
            )  # 跟看涨其实一样，不过还是先写在这里
            vega = (
                S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5
            )  # #跟看涨其实一样，不过还是先写在这里
            theta = (
                -np.exp((b - r) * T) * S * sp.stats.norm.pdf(d1) * sigma / (2 * T**0.5)
                + r * X * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
                + (b - r) * S * np.exp((b - r) * T) * sp.stats.norm.cdf(-d1)
            )
            if b != 0:  # rho比较特别，b是否为0会影响求导结果的形式
                rho = -X * T * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
            else:
                rho = -np.exp(-r * T) * (
                    X * sp.stats.norm.cdf(-d2) - S * sp.stats.norm.cdf(-d1)
                )
        # 写成函数时要有个返回，这里直接把整个写成字典一次性输出。
        greeks = {
            "option_value": option_value,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

        return greeks

    # S = np.linspace(0.1, 200, 100) # 生产0.01到200的100个价格序列
    # result = greeks(CP, S, X, sigma, T, r, b)
    # fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8,12)) # 使用多子图的方式输入结果，所以写的复杂一点
    # greek_list = [['option_value','delta'], ['gamma','vega'], ['theta','rho']] # 和子图的二维数组对应一下
    # for m in range(3):
    #     for n in range(2):
    #         plot_item = greek_list[m][n]
    #         ax[m,n].plot(S,result[plot_item])
    #         ax[m,n].legend([plot_item])
    # plt.show()

    # 3.2、差分方式的实现
    def greeks_diff(
        CP, S, X, sigma, T, r, b, pct_change
    ):  # 计算greeks的函数,差分方式,pct_change表示价格变化的幅度
        option_value = Pri_option.BSM(CP, S, X, sigma, T, r, b)
        delta = (
            Pri_option.BSM(CP, S + S * pct_change, X, sigma, T, r, b)
            - Pri_option.BSM(CP, S - S * pct_change, X, sigma, T, r, b)
        ) / (2 * S * pct_change)
        gamma = (
            Pri_option.BSM(CP, S + S * pct_change, X, sigma, T, r, b)
            + Pri_option.BSM(CP, S - S * pct_change, X, sigma, T, r, b)
            - 2 * Pri_option.BSM(CP, S, X, sigma, T, r, b)
        ) / ((S * pct_change) ** 2)
        vega = (
            Pri_option.BSM(CP, S, X, sigma + sigma * pct_change, T, r, b)
            - Pri_option.BSM(CP, S, X, sigma - sigma * pct_change, T, r, b)
        ) / (2 * sigma * pct_change)
        # theta因为表示的是时间流逝，所+—号是反过来的
        theta = (
            Pri_option.BSM(CP, S, X, sigma, T - T * pct_change, r, b)
            - Pri_option.BSM(CP, S, X, sigma, T + T * pct_change, r, b)
        ) / (2 * T * pct_change)
        if b != 0:
            rho = (
                Pri_option.BSM(
                    CP, S, X, sigma, T, r + r * pct_change, b + r * pct_change
                )
                - Pri_option.BSM(
                    CP, S, X, sigma, T, r - r * pct_change, b - r * pct_change
                )
            ) / (2 * r * pct_change)
        else:
            rho = (
                Pri_option.BSM(CP, S, X, sigma, T, r + r * pct_change, b)
                - Pri_option.BSM(CP, S, X, sigma, T, r - r * pct_change, b)
            ) / (2 * r * pct_change)
        greeks = {
            "option_value": option_value,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

        return greeks

    # S = np.linspace(0.1, 200, 100) # 生产0.01到200的100个价格序列
    # result_diff = greeks_diff(CP, S, X, sigma, T, r, b, pct_change)
    # fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8,12)) # 使用多子图的方式输入结果，所以写的复杂一点
    # greek_list = [['option_value','delta'], ['gamma','vega'], ['theta','rho']] # 和子图的二维数组对应一下
    # for m in range(3):
    #     for n in range(2):
    #         plot_item = greek_list[m][n]
    #         ax[m,n].plot(S, result_diff[plot_item])
    #         ax[m,n].legend([plot_item])
    # plt.show()

    # 4、美式期权定价
    # 4.1、二叉树定价
    def simulate_tree_am(CP, m, S0, T, sigma, K, r, b):  # 二叉树模型美式期权
        """
        CP : 看涨或看跌.
        m : 模拟的期数.
        S0 : 期初价格.
        T : 期限.
        sigma : 波动率.
        K : 行权价格.
        r : 无风险利率.
        b : 持有成本,当b=r时，为标准的无股利模型，b=0时，为black76，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        """
        dt = T / m
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        S = np.zeros((m + 1, m + 1))
        S[0, 0] = S0
        p = (np.exp(b * dt) - d) / (u - d)
        for i in range(1, m + 1):  # 模拟每个节点的价格
            for a in range(i):
                S[a, i] = S[a, i - 1] * u
                S[a + 1, i] = S[a, i - 1] * d
        Sv = np.zeros_like(S)  # 创建期权价值的矩阵，用到从最后一期倒推期权价值
        if CP == "C":
            S_intrinsic = np.maximum(S - K, 0)
        else:
            S_intrinsic = np.maximum(K - S, 0)
        Sv[:, -1] = S_intrinsic[:, -1]
        for i in range(m - 1, -1, -1):  # 反向倒推每个节点的价值
            for a in range(i + 1):
                Sv[a, i] = max(
                    (Sv[a, i + 1] * p + Sv[a + 1, i + 1] * (1 - p)) / np.exp(r * dt),
                    S_intrinsic[a, i],
                )

        return Sv[0, 0]

    # value = simulate_tree_am(CP="C", m=1000, S0=100, K=95, sigma=0.25, T=1, r=0.03, b=0.03)

    # 4.2、BAW公式定价
    # 方法一、论文的迭代方式
    def find_Sx(CP, X, sigma, T, r, b):  # 手动写的标准的牛顿迭代法
        ITERATION_MAX_ERROR = 0.00001  # 牛顿法迭代的精度
        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K = 1 - np.exp(-r * T)
        q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        if CP == "C":
            S_infinite = X / (
                1 - 2 * (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M)) ** -1
            )  # 到期时间为无穷时的价格
            h2 = -(b * T + 2 * sigma * np.sqrt(T)) * X / (S_infinite - X)
            Si = X + (S_infinite - X) * (1 - np.exp(h2))  # 计算种子值
            # print(f"Si的种子值为{Si}")
            LHS = Si - X
            d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            RHS = (
                Pri_option.BSM("C", Si, X, sigma, T, r, b)
                + (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(d1)) * Si / q2
            )
            bi = (
                np.exp((b - r) * T) * sp.stats.norm.cdf(d1) * (1 - 1 / q2)
                + (
                    1
                    - (np.exp((b - r) * T) * sp.stats.norm.pdf(d1)) / sigma / np.sqrt(T)
                )
                / q2
            )  # bi为迭代使用的初始斜率
            while np.abs((LHS - RHS) / X) > ITERATION_MAX_ERROR:
                Si = (X + RHS - bi * Si) / (1 - bi)
                # print(f"Si的值迭代为{Si}")
                LHS = Si - X
                d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
                RHS = (
                    Pri_option.BSM("C", Si, X, sigma, T, r, b)
                    + (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(d1)) * Si / q2
                )
                bi = (
                    np.exp((b - r) * T) * sp.stats.norm.cdf(d1) * (1 - 1 / q2)
                    + (
                        1
                        - (np.exp((b - r) * T) * sp.stats.norm.pdf(d1))
                        / sigma
                        / np.sqrt(T)
                    )
                    / q2
                )
            return Si
        else:
            S_infinite = X / (1 - 2 * (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M)) ** -1)
            h1 = -(b * T - 2 * sigma * np.sqrt(T)) * X / (X - S_infinite)
            Si = S_infinite + (X - S_infinite) * np.exp(h1)  # 计算种子值
            # print(f"Si的种子值为{Si}")
            LHS = X - Si
            d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            RHS = (
                Pri_option.BSM("P", Si, X, sigma, T, r, b)
                - (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(-d1)) * Si / q1
            )
            bi = (
                -np.exp((b - r) * T) * sp.stats.norm.cdf(-d1) * (1 - 1 / q1)
                - (
                    1
                    + (np.exp((b - r) * T) * sp.stats.norm.pdf(-d1))
                    / sigma
                    / np.sqrt(T)
                )
                / q1
            )
            while np.abs((LHS - RHS) / X) > ITERATION_MAX_ERROR:
                Si = (X - RHS + bi * Si) / (1 + bi)
                # print(f"Si的值迭代为{Si}")
                LHS = X - Si
                d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
                RHS = (
                    Pri_option.BSM("P", Si, X, sigma, T, r, b)
                    - (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(-d1)) * Si / q1
                )
                bi = (
                    -np.exp((b - r) * T) * sp.stats.norm.cdf(-d1) * (1 - 1 / q1)
                    - (
                        1
                        + (np.exp((b - r) * T) * sp.stats.norm.pdf(-d1))
                        / sigma
                        / np.sqrt(T)
                    )
                    / q1
                )
            return Si

    # 方法二、使用scipy优化的方式
    def find_Sx_func(CP, S, X, sigma, T, r, b):  # opt版本的迭代
        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K = 1 - np.exp(-r * T)
        q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        if CP == "C":
            LHS = S - X
            RHS = (
                Pri_option.BSM("C", S, X, sigma, T, r, b)
                + (
                    1
                    - np.exp((b - r) * T)
                    * sp.stats.norm.cdf(
                        (np.log(S / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
                    )
                )
                * S
                / q2
            )
            y = (RHS - LHS) ** 2
        else:
            LHS = X - S
            RHS = (
                Pri_option.BSM("P", S, X, sigma, T, r, b)
                - (
                    1
                    - np.exp((b - r) * T)
                    * sp.stats.norm.cdf(
                        -(
                            (np.log(S / X) + (b + sigma**2 / 2) * T)
                            / (sigma * np.sqrt(T))
                        )
                    )
                )
                * S
                / q1
            )
            y = (RHS - LHS) ** 2
        return y

    def find_Sx_opt(CP, S, X, sigma, T, r, b):
        start = S  # 随便给一个S的初始值，或者其他值都行
        func = lambda S: Pri_option.find_Sx_func(CP, S, X, sigma, T, r, b)
        Si = sp.optimize.fmin(func, start)  # 直接做掉包侠
        return Si

    # BAW定价
    def BAW(CP, S, X, sigma, T, r, b, opt_method="newton"):
        if b > r:  # b>r时，美式期权价值和欧式期权相同
            value = Pri_option.BSM(CP, S, X, sigma, T, r, b)

        else:
            M = 2 * r / sigma**2
            N = 2 * b / sigma**2
            K = 1 - np.exp(-r * T)
            if opt_method == "newton":  # 若为牛顿法就用第一种迭代法
                Si = Pri_option.find_Sx(CP, X, sigma, T, r, b)
            else:  # 若不为牛顿法，其他方法这里就是scipy的优化方法
                Si = Pri_option.find_Sx_opt(CP, S, X, sigma, T, r, b)
            d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            if CP == "C":
                q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
                A2 = Si / q2 * (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(d1))
                if S < Si:
                    value = (
                        Pri_option.BSM(CP, S, X, sigma, T, r, b) + A2 * (S / Si) ** q2
                    )
                else:
                    value = S - X

            else:
                q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
                A1 = -Si / q1 * (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(-d1))
                if S > Si:
                    value = (
                        Pri_option.BSM(CP, S, X, sigma, T, r, b) + A1 * (S / Si) ** q1
                    )
                else:
                    value = X - S

        return value

    # result1 = BAW(CP="P", S=100, X=99, sigma=0.2, T=1, r=0.03, b=0, opt_method="newton")
    # result2 = BAW(CP="P", S=100, X=99, sigma=0.2, T=1, r=0.03, b=0, opt_method="scipy")

    # 二分法求解美式期权隐含波动率
    def American_binary(V0, CP, S, X, T, r, b, sigma=0.2):
        """
        Parameters
        ----------
        V0 : 期权价值.
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b=r时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        sigma=0.2 : 预计的初始波动率.
        Returns
        ----------
        返回看涨期权的隐含波动率。
        """
        start = 0  # 初始波动率下限
        end = 2  # 初始波动率上限
        e = 1  # 先给定一个值，让循环运转起来
        while abs(e) >= 0.0001:  # 迭代差异的精度，根据需要调整
            try:
                val = Pri_option.BAW(CP, S, X, sigma, T, r, b, opt_method="newton")
            except ZeroDivisionError:
                print("期权的内在价值大于期权的价格，无法收敛出波动率，会触发除0错误！")
                break
            if val - V0 > 0:  # 若计算的期权价值大于实际价值，说明使用的波动率偏大
                end = sigma
                sigma = (start + end) / 2
                e = end - sigma
            else:  # 若计算的期权价值小于实际价值，说明使用的波动率偏小
                start = sigma
                sigma = (start + end) / 2
                e = start - sigma

        return round(sigma, 4)

    # 4.3、最小二乘蒙特卡洛模拟定价
    def LSM(steps, paths, CP, S0, X, sigma, T, r, b):
        # 代码也可以多写几行计算出所有的提前行权节点，这里为了逻辑清晰就没有列出
        S_path = Pri_option.geo_brownian(steps, paths, T, S0, b, sigma)  # 价格生成路径
        dt = T / steps
        cash_flow = np.zeros_like(S_path)  # 实现创建好现金流量的矩阵，后续使用
        df = np.exp(-r * dt)  # 每一期的折现因子
        if CP == "C":
            cash_flow[-1] = np.maximum(
                S_path[-1] - X, 0
            )  # 先确定最后一期的价值，就是实值额
            exercise_value = np.maximum(S_path - X, 0)
        else:
            cash_flow[-1] = np.maximum(
                X - S_path[-1], 0
            )  # 先确定最后一期的价值，就是实值额
            exercise_value = np.maximum(X - S_path, 0)

        for t in range(steps - 1, 0, -1):  # M-1为倒数第二个时点，从该时点循环至1时点
            df_cash_flow = cash_flow[t + 1] * df
            S_price = S_path[t]  # 标的股价，回归用的X
            itm_index = (
                exercise_value[t] > 0
            )  # 确定实值的index，后面回归要用，通过index的方式可以不破坏价格和现金流矩阵的大小
            reg = np.polyfit(
                S_price[itm_index], df_cash_flow[itm_index], 2
            )  # 实值路径下的标的股价X和下一期的折现现金流Y回归
            holding_value = exercise_value[
                t
            ].copy()  # 创建一个同长度的向量，为了保持index一致，当然也可以用np.zeros_like等方式，本质一样
            holding_value[itm_index] = np.polyval(
                reg, S_price[itm_index]
            )  # 回归出 holding_value，其他的值虽然等于exercise_value，但是后续判断会去除
            ex_index = itm_index & (
                exercise_value[t] > holding_value
            )  # 在实值路径上，进一步寻找出提前行权的index

            df_cash_flow[ex_index] = exercise_value[t][
                ex_index
            ]  # 将cash_flow中提前行权的替换为行权价值，其他保持下一期折现不变
            cash_flow[t] = df_cash_flow

        value = cash_flow[1].mean() * df

        return value

    # LSM(steps=1000, paths=50000, CP="P", S0=40, X=40, sigma=0.2, T=1, r=0.06, b=0.06)

    # 4.4、有限差分法定价
    # 定义一个生成系数矩阵的函数
    def gen_diag(
        M, a, b, c, Sd_idx=0
    ):  # 生成M-1维度的d对角矩阵，a为对角线左下方的值，b为对角的值，c为对角线右上方的值
        """
        Sd_idx代表S划分的下边界Smin/ds的值，一般下边界为0，可设置为0，对于具有下障碍等类型期权需要单独设置Sd_idx
        a,b,c需要函数作为参数，建议使用lambda生成函数
        """
        a_m_1 = [a(i) for i in range(Sd_idx + 2, M)]  # a的系数是从2开始的
        b_m_1 = [b(i) for i in range(Sd_idx + 1, M)]  # b的系数是从1开始的
        c_m_1 = [c(i) for i in range(Sd_idx + 1, M - 1)]  # c的系数是从1开始,但是M-2结束
        diag_matrix = np.diag(a_m_1, -1) + np.diag(b_m_1, 0) + np.diag(c_m_1, 1)

        return diag_matrix

    # 4.4.1、显式有限差分法定价
    def explicit_FD_M(CP, S, K, T, sigma, r, b, M, N):

        ds = (
            S / M
        )  # 确定价格步长，分子用S的意义在于可以让S必定落在网格点上，后续不需要使用插值法
        M = (
            int(K / ds) * 4
        )  # 确定覆盖的价格范围，这里设置为4倍的行权价，也可根据需要设置为其他，这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index，用于方便确定初始S对应的期权价值
        dt = T / N  # 时间步长
        df = 1 / (1 + r * dt)  # 折现因子
        print(f"生产的网格：价格分为M = {M}个点位，时间分为N = {N}个点位")
        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵

        S_array = np.linspace(0, M * ds, M + 1)  # 价格序列
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array  # 生成到期时间的数组，方便后面计算边界条件

        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定终值条件，到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )  # 上边界价格够高，期权表现像远期，这里是远期定价，而不是简单得S-X
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定终值条件，到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        aj = lambda i: 0.5 * (sigma**2 * i**2 - b * i) * dt
        bj = lambda i: 1 - sigma**2 * i**2 * dt
        cj = lambda i: 0.5 * (sigma**2 * i**2 + b * i) * dt
        coef_matrix = Pri_option.gen_diag(M, aj, bj, cj)
        for j in range(N - 1, -1, -1):  # 时间倒推循环
            Z = np.zeros_like(V_grid[1:M, j + 1])  # 用来存储边界条件
            Z[0] = aj(1) * V_grid[0, j + 1]
            Z[-1] = cj(M - 1) * V_grid[-1, j + 1]
            # 矩阵求解
            V_grid[1:M, j] = df * (coef_matrix @ V_grid[1:M, j + 1] + Z)
            # 美式期权提前行权判断
            if CP == "C":
                V_grid[1:M, j] = np.maximum(
                    S_array[1:M] - K, V_grid[1:M, j]
                )  # 美式期权提前行权判断
            else:
                V_grid[1:M, j] = np.maximum(K - S_array[1:M], V_grid[1:M, j])

        return V_grid[S_idx, 0]  # 返回初0时点的初始价格的价值

    # explicit_FD_M(CP="P", S=36, K=40, T=0.5, sigma=0.4, r=0.06, b=0.06, M=125, N=50000)

    # 4.4.2、隐式有限差分法定价
    def implicit_FD(CP, S, K, T, sigma, r, b, M, N):
        """
        隐式有限差分法，比显示更容易收敛，因此N直接指定也容易收敛,但需要通过求逆矩阵解方程组
        f[i+1, j]
        f[i, j]  ➡  f[i, j+1]
        f[i-1, j]
        """
        ds = (
            S / M
        )  # 确定价格步长，用S的意义在于可以让S必定落在网格点上，后续不需要使用插值法
        M = (
            int(K / ds) * 2
        )  # 确定覆盖的价格范围，这里设置为2倍的行权价，也可根据需要设置为其他，这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index，用于方便确定初始S对应的期权价值
        dt = (
            T / N
        )  # 确定步长dt，隐式方法收敛性相对较好，不像显示那么依赖于dt必须得够小，所以这里直接指定N
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array
        print(f"生产的网格：价格分为M = {M}个点位，时间分为N = {N}个点位")

        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵
        S_array = np.linspace(0, M * ds, M + 1)  # 生产价格序列
        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定边界条件，到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定边界条件，到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        # 定义方程的系数的算法，方便后面计算，而且也比较直观
        aj = lambda i: 0.5 * i * (b - sigma**2 * i) * dt
        bj = lambda i: 1 + (r + sigma**2 * i**2) * dt
        cj = lambda i: 0.5 * i * (-b - sigma**2 * i) * dt

        # 用自定义的函数gen_diag有效减少代码
        coefficient_matrix = Pri_option.gen_diag(M, aj, bj, cj)
        M_inverse = np.linalg.inv(coefficient_matrix)

        for j in range(N - 1, -1, -1):  # 隐式也是时间倒推循环，区别在于隐式是要解方程组
            # 准备好解方程组 fj = M**-1 * fj+1,M就是coefficient_matrix的逆矩阵，fj+1的第一项和最后一项需要减去pd*V_grid(0, j)和pu*V_grid(M, j)
            Z = np.zeros_like(V_grid[1:M, j])  # 用来存储边界条件
            Z[0] = aj(1) * V_grid[0, j]  # 隐式这里用的边界条件是j而不是j+1
            Z[-1] = cj(M - 1) * V_grid[-1, j]
            V_grid[1:M, j] = M_inverse @ (V_grid[1:M, j + 1] - Z)

            # 美式期权提前行权判断
            if CP == "C":
                V_grid[1:M, j] = np.maximum(S_array[1:M] - K, V_grid[1:M, j])
            else:
                V_grid[1:M, j] = np.maximum(K - S_array[1:M], V_grid[1:M, j])

        return V_grid[S_idx, 0]  # 返回初0时点的初始价格的价值

    # implicit_FD(CP="P", S=36, K=40, T=0.5, sigma=0.4, r=0.06, b=0.06, M=500, N=2000)

    # 4.4.3、半隐式有限差分法定价
    def CN_FD(CP, S, K, T, sigma, r, b, M, N):
        """
        半隐式有限差分法，Crank_Nicolson，最稳定的方法，推荐
        f[i+1, j]    f[i+1, j+1]
        f[i, j]  ⬅  f[i, j+1]
        f[i-1, j]    f[i-1, j+1]
        """
        ds = (
            S / M
        )  # 确定价格步长，用S的意义在于可以让S必定落在网格点上，后续不需要使用插值法
        M = (
            int(K / ds) * 2
        )  # 确定覆盖的价格范围，这里设置为2倍的行权价，也可根据需要设置为其他，这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index，用于方便确定初始S对应的期权价值
        dt = (
            T / N
        )  # 重新确定步长dt，半隐式方法收敛性相对较好，不像显示那么依赖于dt必须得够小，所以这里直接指定N
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array
        print(f"生产的网格：价格分为M = {M}个点位，时间分为N = {N}个点位")

        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵
        S_array = np.linspace(0, M * ds, M + 1)  # 生产价格序列
        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定边界条件，到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )  # 表现为远期定价
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定边界条件，到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        # 定义方程的系数的算法，方便后面计算，而且也比较直观
        aj = lambda i: 0.25 * (sigma**2 * i**2 - b * i) * dt
        bj = lambda i: -0.5 * (r + sigma**2 * i**2) * dt
        cj = lambda i: 0.25 * (sigma**2 * i**2 + b * i) * dt
        matrix_ones = np.diag([1 for i in range(M - 1)])
        matrix_1 = -Pri_option.gen_diag(M, aj, bj, cj) + matrix_ones
        matrix_2 = Pri_option.gen_diag(M, aj, bj, cj) + matrix_ones
        M1_inverse = np.linalg.inv(matrix_1)

        for j in range(N - 1, -1, -1):  # 隐式也是时间倒推循环，区别在于隐式是要解方程组
            # 准备好解方程组 M_1 * fj = M_2 * fj+1 + b_1
            # Z是对边界条件的处理
            Z = np.zeros_like(V_grid[1:M, j + 1])
            Z[0] = aj(1) * (V_grid[0, j] + V_grid[0, j + 1])
            Z[-1] = cj(M - 1) * (V_grid[-1, j] + V_grid[-1, j + 1])

            V_grid[1:M, j] = M1_inverse @ (matrix_2 @ V_grid[1:M, j + 1] + Z)
            # print(f_j_1,V_grid[1:M,j])
            # 美式期权提前行权判断
            if CP == "C":
                V_grid[1:M, j] = np.maximum(S_array[1:M] - K, V_grid[1:M, j])
            else:
                V_grid[1:M, j] = np.maximum(K - S_array[1:M], V_grid[1:M, j])

        return V_grid[S_idx, 0]  # 返回初0时点的初始价格的价值

    # CN_FD(CP="P", S=36, K=40, T=0.5, sigma=0.4, r=0.06, b=0.06, M=500, N=2000)
