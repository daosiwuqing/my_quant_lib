# 该项目的功能: 量化工具库
import pandas as pd
import numpy as np
import scipy as sp
import re, bisect

import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from package.MyTT import *

# 设置全局字体
fm.fontManager.addfont(
    "C:/Users/29433/AppData/Local/Microsoft/Windows/Fonts/FZHTJW.TTF"
)
mpl.rcParams["font.sans-serif"] = "FZHei-B01S"  # 此处需要用字体文件真正的名称
mpl.rcParams["axes.unicode_minus"] = False  # 解决负号无法显示的问题
font = fm.FontProperties(
    fname="C:/Users/29433/AppData/Local/Microsoft/Windows/Fonts/FZHTJW.TTF"
)


class PriOption(object):
    """
    功能: 封装在期权计算中常用的函数
    """

    # 1, 布朗运动和伊藤引理
    # 1.1, 标准布朗运动
    def standard_brownian(steps: int, paths: int, T: float, S0: float) -> np.ndarray:
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵, 用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数, 当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = S_path[step - 1] + rn[step - 1] * np.sqrt(dt)
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = standard_brownian(steps=100, paths=10, T=1, S0=0)

    # 1.2, 广义的布朗运动
    def brownian(
        steps: int, paths: int, T: float, S0: float, a: float, b: float
    ) -> np.ndarray:
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵, 用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数, 当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = (
                S_path[step - 1] + a * dt + b * rn[step - 1] * np.sqrt(dt)
            )  # 和标准布朗运动的区别就在这一行
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = brownian(steps=100, paths=10, T=1, S0=0, a=5, b=2)

    # 1.3, 几何布朗运动
    def geo_brownian(
        steps: int, paths: int, T: float, S0: float, u: float, sigma: float
    ) -> np.ndarray:
        dt = T / steps  # 求出dt
        S_path = np.zeros((steps + 1, paths))  # 创建一个矩阵, 用来准备储存模拟情况
        S_path[0] = S0  # 起点设置
        rn = np.random.standard_normal(
            S_path.shape
        )  # 一次性创建出需要的正态分布随机数, 当然也可以写在循环里每次创建一个时刻的随机数
        for step in range(1, steps + 1):
            S_path[step] = S_path[step - 1] * np.exp(
                (u - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rn[step]
            )  # 和其他布朗运动的区别就在这一行
        plt.plot(S_path)
        plt.show()

        return S_path

    # S_path = geo_brownian(steps=100, paths=50, T=1, S0=100, u=0.03, sigma=0.2)

    # 2, BSM公式
    # 2.1, 定价公式
    def BSM(
        CP: str, S: float, X: float, sigma: float, T: float, r: float, b: float
    ) -> float:
        """
        Parameters
        ----------
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        sigma : 波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本, 当b=r时, 为标准的无股利模型, b=0时, 为期货期权, b为r-q时, 为支付股利模型, b为r-rf时为外汇期权.
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

    # 2.2, 二分法求解隐含波动率
    def binary(
        V0: float,
        CP: str,
        S: float,
        X: float,
        T: float,
        r: float,
        b: float,
        vol_est: float | None = None,
    ) -> float:
        """
        Parameters
        ----------
        V0 : 期权价值.
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本, 当b=r时, 为标准的无股利模型, b=0时, 为期货期权, b为r-q时, 为支付股利模型, b为r-rf时为外汇期权.
        vol_est : 预计的初始波动率.
        Returns
        ----------
        返回看涨期权的隐含波动率。
        """
        if vol_est is None:
            vol_est = 0.2

        start = 0  # 初始波动率下限
        end = 2  # 初始波动率上限
        e = 1  # 先给定一个值, 让循环运转起来
        while abs(e) >= 0.0001:  # 迭代差异的精度, 根据需要调整
            try:
                val = PriOption.BSM(CP, S, X, vol_est, T, r, b)
            except ZeroDivisionError:
                print("期权的内在价值大于期权的价格, 无法收敛出波动率, 会触发除0错误！")
                break
            if val - V0 > 0:  # 若计算的期权价值大于实际价值, 说明使用的波动率偏大
                end = vol_est
                vol_est = (start + end) / 2
                e = end - vol_est
            else:  # 若计算的期权价值小于实际价值, 说明使用的波动率偏小
                start = vol_est
                vol_est = (start + end) / 2
                e = start - vol_est

        return round(vol_est, 4)

    # value = BSM(CP="C", S=100, X=95, T=1, sigma=0.25, r=0.03, b=0.03) #实验一个期权的价值
    # print(value)
    # vol = binary(V0=value, CP="C", S=100, X=95, T=1, r=0.03, b=0.03)  #根据刚才实验的期权价值求一下波动率是否正确
    # print(vol)

    # 2.3 牛顿法求解隐含波动率
    def newton(
        V0: float,
        CP: str,
        S: float,
        X: float,
        T: float,
        r: float,
        b: float,
        vol_est: float | None = None,
        n_iter: int | None = None,
    ) -> float:
        if vol_est is None:
            vol_est = 0.25
        if n_iter is None:  # n_iter表示迭代的次数
            n_iter = 1000

        for i in range(n_iter):
            d1 = (np.log(S / X) + (b + vol_est**2 / 2) * T) / (vol_est * np.sqrt(T))
            vega = S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5  # 计算vega
            vol_est = (
                vol_est - (PriOption.BSM(CP, S, X, vol_est, T, r, b) - V0) / vega
            )  # 每次迭代都重新算一下波动率

        return vol_est

    # vol = newton(V0, CP, S, X, T, r, b, vol_est=0.2, n_iter=1000)
    # print(vol)

    # 3, 欧式期权希腊字母计算
    # 3.1, 解析解下的实现
    def greeks(
        CP: str, S: float, X: float, sigma: float, T: float, r: float, b: float
    ) -> dict:  # 计算greeks的函数
        """
        Parameters
        ----------
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        sigma : 波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本, 当b=r时, 为标准的无股利模型, b=0时, 为期货期权, b为r-q时, 为支付股利模型, b为r-rf时为外汇期权.
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
            )  # 注意是pdf, 概率密度函数
            vega = S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5  # 计算vega
            theta = (
                -np.exp((b - r) * T) * S * sp.stats.norm.pdf(d1) * sigma / (2 * T**0.5)
                - r * X * np.exp(-r * T) * sp.stats.norm.cdf(d2)
                - (b - r) * S * np.exp((b - r) * T) * sp.stats.norm.cdf(d1)
            )
            if b != 0:  # rho比较特别, b是否为0会影响求导结果的形式
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
            )  # 跟看涨其实一样, 不过还是先写在这里
            vega = (
                S * np.exp((b - r) * T) * sp.stats.norm.pdf(d1) * T**0.5
            )  # #跟看涨其实一样, 不过还是先写在这里
            theta = (
                -np.exp((b - r) * T) * S * sp.stats.norm.pdf(d1) * sigma / (2 * T**0.5)
                + r * X * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
                + (b - r) * S * np.exp((b - r) * T) * sp.stats.norm.cdf(-d1)
            )
            if b != 0:  # rho比较特别, b是否为0会影响求导结果的形式
                rho = -X * T * np.exp(-r * T) * sp.stats.norm.cdf(-d2)
            else:
                rho = -np.exp(-r * T) * (
                    X * sp.stats.norm.cdf(-d2) - S * sp.stats.norm.cdf(-d1)
                )
        # 写成函数时要有个返回, 这里直接把整个写成字典一次性输出。
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
    # fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8,12)) # 使用多子图的方式输入结果, 所以写的复杂一点
    # greek_list = [['option_value','delta'], ['gamma','vega'], ['theta','rho']] # 和子图的二维数组对应一下
    # for m in range(3):
    #     for n in range(2):
    #         plot_item = greek_list[m][n]
    #         ax[m,n].plot(S,result[plot_item])
    #         ax[m,n].legend([plot_item])
    # plt.show()

    # 3.2, 差分方式的实现
    def greeks_diff(
        CP: str,
        S: float,
        X: float,
        sigma: float,
        T: float,
        r: float,
        b: float,
        pct_change: float,
    ) -> dict:  # 计算greeks的函数,差分方式,pct_change表示价格变化的幅度
        option_value = PriOption.BSM(CP, S, X, sigma, T, r, b)
        delta = (
            PriOption.BSM(CP, S + S * pct_change, X, sigma, T, r, b)
            - PriOption.BSM(CP, S - S * pct_change, X, sigma, T, r, b)
        ) / (2 * S * pct_change)
        gamma = (
            PriOption.BSM(CP, S + S * pct_change, X, sigma, T, r, b)
            + PriOption.BSM(CP, S - S * pct_change, X, sigma, T, r, b)
            - 2 * PriOption.BSM(CP, S, X, sigma, T, r, b)
        ) / ((S * pct_change) ** 2)
        vega = (
            PriOption.BSM(CP, S, X, sigma + sigma * pct_change, T, r, b)
            - PriOption.BSM(CP, S, X, sigma - sigma * pct_change, T, r, b)
        ) / (2 * sigma * pct_change)
        # theta因为表示的是时间流逝, 所+—号是反过来的
        theta = (
            PriOption.BSM(CP, S, X, sigma, T - T * pct_change, r, b)
            - PriOption.BSM(CP, S, X, sigma, T + T * pct_change, r, b)
        ) / (2 * T * pct_change)
        if b != 0:
            rho = (
                PriOption.BSM(
                    CP, S, X, sigma, T, r + r * pct_change, b + r * pct_change
                )
                - PriOption.BSM(
                    CP, S, X, sigma, T, r - r * pct_change, b - r * pct_change
                )
            ) / (2 * r * pct_change)
        else:
            rho = (
                PriOption.BSM(CP, S, X, sigma, T, r + r * pct_change, b)
                - PriOption.BSM(CP, S, X, sigma, T, r - r * pct_change, b)
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
    # fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(8,12)) # 使用多子图的方式输入结果, 所以写的复杂一点
    # greek_list = [['option_value','delta'], ['gamma','vega'], ['theta','rho']] # 和子图的二维数组对应一下
    # for m in range(3):
    #     for n in range(2):
    #         plot_item = greek_list[m][n]
    #         ax[m,n].plot(S, result_diff[plot_item])
    #         ax[m,n].legend([plot_item])
    # plt.show()

    # 4, 美式期权定价
    # 4.1, 二叉树定价
    def simulate_tree_am(
        CP: str, m: int, S0: float, T: float, sigma: float, K: float, r: float, b: float
    ) -> float:  # 二叉树模型美式期权
        """
        CP : 看涨或看跌.
        m : 模拟的期数.
        S0 : 期初价格.
        T : 期限.
        sigma : 波动率.
        K : 行权价格.
        r : 无风险利率.
        b : 持有成本,当b=r时, 为标准的无股利模型, b=0时, 为black76, b为r-q时, 为支付股利模型, b为r-rf时为外汇期权.
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
        Sv = np.zeros_like(S)  # 创建期权价值的矩阵, 用到从最后一期倒推期权价值
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

    # 4.2, BAW公式定价
    # 方法一, 论文的迭代方式
    def _find_Sx(
        CP: str, X: float, sigma: float, T: float, r: float, b: float
    ) -> float:  # 手动写的标准的牛顿迭代法
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
                PriOption.BSM("C", Si, X, sigma, T, r, b)
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
                    PriOption.BSM("C", Si, X, sigma, T, r, b)
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
                PriOption.BSM("P", Si, X, sigma, T, r, b)
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
                    PriOption.BSM("P", Si, X, sigma, T, r, b)
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

    # 方法二, 使用scipy优化的方式
    def _find_Sx_func(
        CP: str, S: float, X: float, sigma: float, T: float, r: float, b: float
    ) -> float:  # opt版本的迭代
        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K = 1 - np.exp(-r * T)
        q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
        if CP == "C":
            LHS = S - X
            RHS = (
                PriOption.BSM("C", S, X, sigma, T, r, b)
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
                PriOption.BSM("P", S, X, sigma, T, r, b)
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

    def _find_Sx_opt(
        CP: str, S: float, X: float, sigma: float, T: float, r: float, b: float
    ) -> float:
        start = S  # 随便给一个S的初始值, 或者其他值都行
        func = lambda S: PriOption._find_Sx_func(CP, S, X, sigma, T, r, b)
        Si = sp.optimize.fmin(func, start)  # 直接做掉包侠
        return Si

    # BAW定价
    def BAW(
        CP: str,
        S: float,
        X: float,
        sigma: float,
        T: float,
        r: float,
        b: float,
        opt_method: str | None = None,
    ):
        if opt_method is None:
            opt_method = "newton"

        if b > r:  # b>r时, 美式期权价值和欧式期权相同
            value = PriOption.BSM(CP, S, X, sigma, T, r, b)

        else:
            M = 2 * r / sigma**2
            N = 2 * b / sigma**2
            K = 1 - np.exp(-r * T)
            if opt_method == "newton":  # 若为牛顿法就用第一种迭代法
                Si = PriOption._find_Sx(CP, X, sigma, T, r, b)
            else:  # 若不为牛顿法, 其他方法这里就是scipy的优化方法
                Si = PriOption._find_Sx_opt(CP, S, X, sigma, T, r, b)
            d1 = (np.log(Si / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            if CP == "C":
                q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
                A2 = Si / q2 * (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(d1))
                if S < Si:
                    value = (
                        PriOption.BSM(CP, S, X, sigma, T, r, b) + A2 * (S / Si) ** q2
                    )
                else:
                    value = S - X

            else:
                q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2
                A1 = -Si / q1 * (1 - np.exp((b - r) * T) * sp.stats.norm.cdf(-d1))
                if S > Si:
                    value = (
                        PriOption.BSM(CP, S, X, sigma, T, r, b) + A1 * (S / Si) ** q1
                    )
                else:
                    value = X - S

        return value

    # result1 = BAW(CP="P", S=100, X=99, sigma=0.2, T=1, r=0.03, b=0, opt_method="newton")
    # result2 = BAW(CP="P", S=100, X=99, sigma=0.2, T=1, r=0.03, b=0, opt_method="scipy")

    # 二分法求解美式期权隐含波动率
    def American_binary(
        V0: float,
        CP: str,
        S: float,
        X: float,
        T: float,
        r: float,
        b: float,
        sigma: float | None = None,
    ) -> float:
        """
        Parameters
        ----------
        V0 : 期权价值.
        CP : 看涨或看跌"C"or"P".
        S : 标的价格.
        X : 行权价格.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本, 当b=r时, 为标准的无股利模型, b=0时, 为期货期权, b为r-q时, 为支付股利模型, b为r-rf时为外汇期权.
        sigma=0.2 : 预计的初始波动率.
        Returns
        ----------
        返回看涨期权的隐含波动率。
        """
        if sigma is None:
            sigma = 0.2

        start = 0  # 初始波动率下限
        end = 2  # 初始波动率上限
        e = 1  # 先给定一个值, 让循环运转起来
        while abs(e) >= 0.0001:  # 迭代差异的精度, 根据需要调整
            try:
                val = PriOption.BAW(CP, S, X, sigma, T, r, b, opt_method="newton")
            except ZeroDivisionError:
                print("期权的内在价值大于期权的价格, 无法收敛出波动率, 会触发除0错误！")
                break
            if val - V0 > 0:  # 若计算的期权价值大于实际价值, 说明使用的波动率偏大
                end = sigma
                sigma = (start + end) / 2
                e = end - sigma
            else:  # 若计算的期权价值小于实际价值, 说明使用的波动率偏小
                start = sigma
                sigma = (start + end) / 2
                e = start - sigma

        return round(sigma, 4)

    # 4.3, 最小二乘蒙特卡洛模拟定价
    def LSM(
        steps: int,
        paths: int,
        CP: str,
        S0: float,
        X: float,
        sigma: float,
        T: float,
        r: float,
        b: float,
    ) -> float:
        # 代码也可以多写几行计算出所有的提前行权节点, 这里为了逻辑清晰就没有列出
        S_path = PriOption.geo_brownian(steps, paths, T, S0, b, sigma)  # 价格生成路径
        dt = T / steps
        cash_flow = np.zeros_like(S_path)  # 实现创建好现金流量的矩阵, 后续使用
        df = np.exp(-r * dt)  # 每一期的折现因子
        if CP == "C":
            cash_flow[-1] = np.maximum(
                S_path[-1] - X, 0
            )  # 先确定最后一期的价值, 就是实值额
            exercise_value = np.maximum(S_path - X, 0)
        else:
            cash_flow[-1] = np.maximum(
                X - S_path[-1], 0
            )  # 先确定最后一期的价值, 就是实值额
            exercise_value = np.maximum(X - S_path, 0)

        for t in range(steps - 1, 0, -1):  # M-1为倒数第二个时点, 从该时点循环至1时点
            df_cash_flow = cash_flow[t + 1] * df
            S_price = S_path[t]  # 标的股价, 回归用的X
            itm_index = (
                exercise_value[t] > 0
            )  # 确定实值的index, 后面回归要用, 通过index的方式可以不破坏价格和现金流矩阵的大小
            reg = np.polyfit(
                S_price[itm_index], df_cash_flow[itm_index], 2
            )  # 实值路径下的标的股价X和下一期的折现现金流Y回归
            holding_value = exercise_value[
                t
            ].copy()  # 创建一个同长度的向量, 为了保持index一致, 当然也可以用np.zeros_like等方式, 本质一样
            holding_value[itm_index] = np.polyval(
                reg, S_price[itm_index]
            )  # 回归出 holding_value, 其他的值虽然等于exercise_value, 但是后续判断会去除
            ex_index = itm_index & (
                exercise_value[t] > holding_value
            )  # 在实值路径上, 进一步寻找出提前行权的index

            df_cash_flow[ex_index] = exercise_value[t][
                ex_index
            ]  # 将cash_flow中提前行权的替换为行权价值, 其他保持下一期折现不变
            cash_flow[t] = df_cash_flow

        value = cash_flow[1].mean() * df

        return value

    # LSM(steps=1000, paths=50000, CP="P", S0=40, X=40, sigma=0.2, T=1, r=0.06, b=0.06)

    # 4.4, 有限差分法定价
    # 定义一个生成系数矩阵的函数
    def _gen_diag(
        M: int, a: float, b: float, c: float, Sd_idx: int | None = None
    ) -> (
        np.ndarray
    ):  # 生成M-1维度的d对角矩阵, a为对角线左下方的值, b为对角的值, c为对角线右上方的值
        """
        Sd_idx代表S划分的下边界Smin/ds的值, 一般下边界为0, 可设置为0, 对于具有下障碍等类型期权需要单独设置Sd_idx
        a,b,c需要函数作为参数, 建议使用lambda生成函数
        """
        if Sd_idx is None:
            Sd_idx = 0

        a_m_1 = [a(i) for i in range(Sd_idx + 2, M)]  # a的系数是从2开始的
        b_m_1 = [b(i) for i in range(Sd_idx + 1, M)]  # b的系数是从1开始的
        c_m_1 = [c(i) for i in range(Sd_idx + 1, M - 1)]  # c的系数是从1开始,但是M-2结束
        diag_matrix = np.diag(a_m_1, -1) + np.diag(b_m_1, 0) + np.diag(c_m_1, 1)

        return diag_matrix

    # 4.4.1, 显式有限差分法定价
    def explicit_FD_M(
        CP: str,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float,
        b: float,
        M: int,
        N: int,
    ) -> float:
        ds = (
            S / M
        )  # 确定价格步长, 分子用S的意义在于可以让S必定落在网格点上, 后续不需要使用插值法
        M = (
            int(K / ds) * 4
        )  # 确定覆盖的价格范围, 这里设置为4倍的行权价, 也可根据需要设置为其他, 这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index, 用于方便确定初始S对应的期权价值
        dt = T / N  # 时间步长
        df = 1 / (1 + r * dt)  # 折现因子
        print(f"生产的网格: 价格分为M = {M}个点位, 时间分为N = {N}个点位")
        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵

        S_array = np.linspace(0, M * ds, M + 1)  # 价格序列
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array  # 生成到期时间的数组, 方便后面计算边界条件

        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定终值条件, 到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )  # 上边界价格够高, 期权表现像远期, 这里是远期定价, 而不是简单得S-X
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定终值条件, 到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        aj = lambda i: 0.5 * (sigma**2 * i**2 - b * i) * dt
        bj = lambda i: 1 - sigma**2 * i**2 * dt
        cj = lambda i: 0.5 * (sigma**2 * i**2 + b * i) * dt
        coef_matrix = PriOption._gen_diag(M, aj, bj, cj)
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

    # 4.4.2, 隐式有限差分法定价
    def implicit_FD(
        CP: str,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float,
        b: float,
        M: int,
        N: int,
    ) -> float:
        """
        隐式有限差分法, 比显示更容易收敛, 因此N直接指定也容易收敛,但需要通过求逆矩阵解方程组
        f[i+1, j]
        f[i, j]  ➡  f[i, j+1]
        f[i-1, j]
        """
        ds = (
            S / M
        )  # 确定价格步长, 用S的意义在于可以让S必定落在网格点上, 后续不需要使用插值法
        M = (
            int(K / ds) * 2
        )  # 确定覆盖的价格范围, 这里设置为2倍的行权价, 也可根据需要设置为其他, 这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index, 用于方便确定初始S对应的期权价值
        dt = (
            T / N
        )  # 确定步长dt, 隐式方法收敛性相对较好, 不像显示那么依赖于dt必须得够小, 所以这里直接指定N
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array
        print(f"生产的网格: 价格分为M = {M}个点位, 时间分为N = {N}个点位")

        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵
        S_array = np.linspace(0, M * ds, M + 1)  # 生产价格序列
        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定边界条件, 到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定边界条件, 到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        # 定义方程的系数的算法, 方便后面计算, 而且也比较直观
        aj = lambda i: 0.5 * i * (b - sigma**2 * i) * dt
        bj = lambda i: 1 + (r + sigma**2 * i**2) * dt
        cj = lambda i: 0.5 * i * (-b - sigma**2 * i) * dt

        # 用自定义的函数gen_diag有效减少代码
        coefficient_matrix = PriOption._gen_diag(M, aj, bj, cj)
        M_inverse = np.linalg.inv(coefficient_matrix)

        for j in range(N - 1, -1, -1):  # 隐式也是时间倒推循环, 区别在于隐式是要解方程组
            # 准备好解方程组 fj = M**-1 * fj+1,M就是coefficient_matrix的逆矩阵, fj+1的第一项和最后一项需要减去pd*V_grid(0, j)和pu*V_grid(M, j)
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

    # 4.4.3, 半隐式有限差分法定价
    def CN_FD(
        CP: str,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: float,
        b: float,
        M: int,
        N: int,
    ) -> float:
        """
        半隐式有限差分法, Crank_Nicolson, 最稳定的方法, 推荐
        f[i+1, j]    f[i+1, j+1]
        f[i, j]  ⬅  f[i, j+1]
        f[i-1, j]    f[i-1, j+1]
        """
        ds = (
            S / M
        )  # 确定价格步长, 用S的意义在于可以让S必定落在网格点上, 后续不需要使用插值法
        M = (
            int(K / ds) * 2
        )  # 确定覆盖的价格范围, 这里设置为2倍的行权价, 也可根据需要设置为其他, 这里根据价格范围重新计算价格点位数量M
        S_idx = int(S / ds)  # S所在的index, 用于方便确定初始S对应的期权价值
        dt = (
            T / N
        )  # 重新确定步长dt, 半隐式方法收敛性相对较好, 不像显示那么依赖于dt必须得够小, 所以这里直接指定N
        T_array = np.linspace(0, N * dt, N + 1)  # 时间序列
        T2M_array = T_array[-1] - T_array
        print(f"生产的网格: 价格分为M = {M}个点位, 时间分为N = {N}个点位")

        V_grid = np.zeros((M + 1, N + 1))  # 预先生成包括0在内的期权价值矩阵
        S_array = np.linspace(0, M * ds, M + 1)  # 生产价格序列
        if CP == "C":
            V_grid[:, N] = np.maximum(
                S_array - K, 0
            )  # 确定边界条件, 到期时期权价值很好计算
            V_grid[M] = np.exp(-r * T2M_array) * (
                S_array[-1] * np.exp(b * T2M_array) - K
            )  # 表现为远期定价
        else:
            V_grid[:, N] = np.maximum(
                K - S_array, 0
            )  # 确定边界条件, 到期时期权价值很好计算
            V_grid[0] = np.exp(-r * T2M_array) * K

        # 定义方程的系数的算法, 方便后面计算, 而且也比较直观
        aj = lambda i: 0.25 * (sigma**2 * i**2 - b * i) * dt
        bj = lambda i: -0.5 * (r + sigma**2 * i**2) * dt
        cj = lambda i: 0.25 * (sigma**2 * i**2 + b * i) * dt
        matrix_ones = np.diag([1 for i in range(M - 1)])
        matrix_1 = -PriOption._gen_diag(M, aj, bj, cj) + matrix_ones
        matrix_2 = PriOption._gen_diag(M, aj, bj, cj) + matrix_ones
        M1_inverse = np.linalg.inv(matrix_1)

        for j in range(N - 1, -1, -1):  # 隐式也是时间倒推循环, 区别在于隐式是要解方程组
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


class GenerateIndex(object):
    """
    功能: 构建各种类型的指数
    """

    def generate_main_price_index(df1: pd.DataFrame) -> pd.DataFrame:
        """
        功能: 根据日频行情数据构建出主力连续合约价格指数

        参数:
            df1: 含有calender_day, instrument_id, close, volume, turnover, open_interest等列

        返回:
            pd.DataFrame: 含有calender_day, instrument_id, close, volume, total_volume, turnover, total_turnover, open_interest, total_open_interest等列
        """
        df1 = df1.dropna(
            subset=["close", "volume", "turnover", "open_interest"], how="any"
        )
        df1 = df1.sort_values(
            by=["calender_day", "open_interest"], ascending=[True, False]
        )

        df_list = []
        pre_main_contract = ""
        for calender_day in sorted(set(df1["calender_day"])):
            df2 = df1[df1.calender_day == calender_day]
            if not pre_main_contract:  # 如果没有主力合约, 就要建立一个主力合约
                main_contract = df2.iloc[0]["instrument_id"]
                main_contract_close = df2.iloc[0]["close"]
                main_contract_volume = df2.iloc[0]["volume"]
                total_volume = sum(df2["volume"])
                main_contract_turnover = df2.iloc[0]["turnover"]
                total_turnover = sum(df2["turnover"])
                main_contract_open_interest = df2.iloc[0]["open_interest"]
                total_open_interest = sum(df2["open_interest"])
                df_list.append(
                    [
                        calender_day,
                        main_contract,
                        main_contract_close,
                        main_contract_volume,
                        total_volume,
                        main_contract_turnover,
                        total_turnover,
                        main_contract_open_interest,
                        total_open_interest,
                    ]
                )
                pre_main_contract = main_contract
            else:  # 如果已经有主力合约了, 就要判断是否展期
                if (
                    OtherTools.cal_date_spread(
                        pd.Timestamp(calender_day),
                        OtherTools.get_maturity_date(
                            pre_main_contract, pd.Timestamp(calender_day)
                        ),
                        pd.to_datetime(sorted(set(df1["calender_day"]))),
                    )["months_remaining"]
                ) <= 1:  # 如果合约离到期日小于等于1个月 (不考虑天数) , 就要强展
                    roll_df = df2[
                        df2["instrument_id"].apply(
                            lambda x: (
                                True
                                if OtherTools.get_maturity_date(
                                    x, pd.Timestamp(calender_day)
                                )
                                > OtherTools.get_maturity_date(
                                    pre_main_contract, pd.Timestamp(calender_day)
                                )
                                else False
                            )
                        )
                    ]  # 筛选出比当前主力合约更晚到期的合约

                    main_contract = roll_df.iloc[0]["instrument_id"]
                    main_contract_close = roll_df.iloc[0]["close"]
                    main_contract_volume = roll_df.iloc[0]["volume"]
                    total_volume = sum(df2["volume"])
                    main_contract_turnover = roll_df.iloc[0]["turnover"]
                    total_turnover = sum(df2["turnover"])
                    main_contract_open_interest = roll_df.iloc[0]["open_interest"]
                    total_open_interest = sum(df2["open_interest"])
                    df_list.append(
                        [
                            calender_day,
                            main_contract,
                            main_contract_close,
                            main_contract_volume,
                            total_volume,
                            main_contract_turnover,
                            total_turnover,
                            main_contract_open_interest,
                            total_open_interest,
                        ]
                    )
                    pre_main_contract = main_contract
                else:  # 无需强展, 就要判断是否自然展期
                    largest_open_interest_contract = df2.iloc[0]["instrument_id"]
                    if OtherTools.get_maturity_date(
                        largest_open_interest_contract, pd.Timestamp(calender_day)
                    ) > OtherTools.get_maturity_date(
                        pre_main_contract, pd.Timestamp(calender_day)
                    ):  # 如果最大交易量的合约比主力合约更晚到期
                        main_contract = largest_open_interest_contract
                        main_contract_close = df2.iloc[0]["close"]
                        main_contract_volume = df2.iloc[0]["volume"]
                        total_volume = sum(df2["volume"])
                        main_contract_turnover = df2.iloc[0]["turnover"]
                        total_turnover = sum(df2["turnover"])
                        main_contract_open_interest = df2.iloc[0]["open_interest"]
                        total_open_interest = sum(df2["open_interest"])
                        df_list.append(
                            [
                                calender_day,
                                main_contract,
                                main_contract_close,
                                main_contract_volume,
                                total_volume,
                                main_contract_turnover,
                                total_turnover,
                                main_contract_open_interest,
                                total_open_interest,
                            ]
                        )
                        pre_main_contract = main_contract
                    else:  # 不展期
                        stay_df = df2[df2.instrument_id == pre_main_contract]
                        main_contract = stay_df.iloc[0]["instrument_id"]
                        main_contract_close = stay_df.iloc[0]["close"]
                        main_contract_volume = stay_df.iloc[0]["volume"]
                        total_volume = sum(df2["volume"])
                        main_contract_turnover = stay_df.iloc[0]["turnover"]
                        total_turnover = sum(df2["turnover"])
                        main_contract_open_interest = stay_df.iloc[0]["open_interest"]
                        total_open_interest = sum(df2["open_interest"])
                        df_list.append(
                            [
                                calender_day,
                                main_contract,
                                main_contract_close,
                                main_contract_volume,
                                total_volume,
                                main_contract_turnover,
                                total_turnover,
                                main_contract_open_interest,
                                total_open_interest,
                            ]
                        )
        result = pd.DataFrame(
            df_list,
            columns=[
                "calender_day",
                "instrument_id",
                "close",
                "volume",
                "total_volume",
                "turnover",
                "total_turnover",
                "open_interest",
                "total_open_interest",
            ],
        )

        return result

    def generate_weight_price_index(df1: pd.DataFrame) -> pd.DataFrame:
        """
        功能: 根据日频行情数据构建出成交量加权价格指数

        参数:
            df1: 含有calender_day, instrument_id, close, volume, turnover, open_interest等列

        返回:
            pd.DataFrame: 含有calender_day, price, volume, turnover, open_interest等列
        """
        df1 = df1.dropna(
            subset=["close", "volume", "turnover", "open_interest"], how="any"
        )
        df1 = df1.sort_values(
            by=["calender_day", "open_interest"], ascending=[True, False]
        )

        df_list = []
        for calender_day in sorted(set(df1["calender_day"])):
            df2 = df1[df1.calender_day == calender_day]
            sum_volume = sum(df2["volume"])
            sum_turnover = sum(df2["turnover"])
            sum_open_interest = sum(df2["open_interest"])
            price = sum(df2["close"] * (df2["volume"] / sum_volume))
            df_list.append(
                [calender_day, price, sum_volume, sum_turnover, sum_open_interest]
            )
        result = pd.DataFrame(
            df_list,
            columns=["calender_day", "price", "volume", "turnover", "open_interest"],
        )

        return result

    def generate_return_index(df1: pd.DataFrame) -> pd.DataFrame:
        """
        功能: 根据日频行情数据构建出主力连续合约收益率指数

        参数:
            df1: 含有calender_day, instrument_id, close, volume, turnover, open_interest等列

        返回:
            pd.DataFrame: 含有calender_day, instrument_id, close, volume, total_volume, turnover, total_turnover, open_interest, total_open_interest, index等列
        """
        df1 = df1.dropna(
            subset=["close", "volume", "turnover", "open_interest"], how="any"
        )
        df1 = df1.sort_values(
            by=["calender_day", "open_interest"], ascending=[True, False]
        )

        df_list = []
        pre_main_contract = ""
        pre_close = 0
        pre_index = 0
        for calender_day in sorted(set(df1["calender_day"])):
            df2 = df1[df1.calender_day == calender_day]
            if pre_main_contract == "":  # 如果没有主力合约, 就要建立一个主力合约
                main_contract = df2.iloc[0]["instrument_id"]
                main_contract_close = df2.iloc[0]["close"]
                main_contract_volume = df2.iloc[0]["volume"]
                total_volume = sum(df2["volume"])
                main_contract_turnover = df2.iloc[0]["turnover"]
                total_turnover = sum(df2["turnover"])
                main_contract_open_interest = df2.iloc[0]["open_interest"]
                total_open_interest = sum(df2["open_interest"])
                index = 1000
                df_list.append(
                    [
                        calender_day,
                        main_contract,
                        main_contract_close,
                        main_contract_volume,
                        total_volume,
                        main_contract_turnover,
                        total_turnover,
                        main_contract_open_interest,
                        total_open_interest,
                        index,
                    ]
                )
                pre_main_contract = main_contract
                pre_close = main_contract_close
                pre_index = index
            else:  # 如果已经有主力合约了, 就要判断是否展期
                if (
                    OtherTools.cal_date_spread(
                        pd.Timestamp(calender_day),
                        OtherTools.get_maturity_date(
                            pre_main_contract, pd.Timestamp(calender_day)
                        ),
                        pd.to_datetime(sorted(set(df1["calender_day"]))),
                    )["months_remaining"]
                ) <= 1:  # 如果合约离到期日小于等于1个月 (不考虑天数) , 就要强展
                    roll_df = df2[
                        df2["instrument_id"].apply(
                            lambda x: (
                                True
                                if OtherTools.get_maturity_date(
                                    x, pd.Timestamp(calender_day)
                                )
                                > OtherTools.get_maturity_date(
                                    pre_main_contract, pd.Timestamp(calender_day)
                                )
                                else False
                            )
                        )
                    ]  # 筛选出比当前主力合约更晚到期的合约

                    main_contract = roll_df.iloc[0]["instrument_id"]
                    main_contract_close = roll_df.iloc[0]["close"]
                    main_contract_volume = roll_df.iloc[0]["volume"]
                    total_volume = sum(df2["volume"])
                    main_contract_turnover = roll_df.iloc[0]["turnover"]
                    total_turnover = sum(df2["turnover"])
                    main_contract_open_interest = roll_df.iloc[0]["open_interest"]
                    total_open_interest = sum(df2["open_interest"])
                    index = pre_index * (
                        df2.loc[
                            df2["instrument_id"] == pre_main_contract, "close"
                        ].values[0]
                        / pre_close
                    )  # 收盘时才进行展期操作
                    df_list.append(
                        [
                            calender_day,
                            main_contract,
                            main_contract_close,
                            main_contract_volume,
                            total_volume,
                            main_contract_turnover,
                            total_turnover,
                            main_contract_open_interest,
                            total_open_interest,
                            index,
                        ]
                    )
                    pre_main_contract = main_contract
                    pre_close = main_contract_close
                    pre_index = index
                else:  # 无需强展, 就要判断是否自然展期
                    largest_open_interest_contract = df2.iloc[0]["instrument_id"]
                    if OtherTools.get_maturity_date(
                        largest_open_interest_contract, pd.Timestamp(calender_day)
                    ) > OtherTools.get_maturity_date(
                        pre_main_contract, pd.Timestamp(calender_day)
                    ):  # 如果最大交易量的合约比主力合约更晚到期
                        main_contract = largest_open_interest_contract
                        main_contract_close = df2.iloc[0]["close"]
                        main_contract_volume = df2.iloc[0]["volume"]
                        total_volume = sum(df2["volume"])
                        main_contract_turnover = df2.iloc[0]["turnover"]
                        total_turnover = sum(df2["turnover"])
                        main_contract_open_interest = df2.iloc[0]["open_interest"]
                        total_open_interest = sum(df2["open_interest"])
                        index = pre_index * (
                            df2.loc[
                                df2["instrument_id"] == pre_main_contract, "close"
                            ].values[0]
                            / pre_close
                        )  # 收盘时才进行展期操作
                        df_list.append(
                            [
                                calender_day,
                                main_contract,
                                main_contract_close,
                                main_contract_volume,
                                total_volume,
                                main_contract_turnover,
                                total_turnover,
                                main_contract_open_interest,
                                total_open_interest,
                                index,
                            ]
                        )
                        pre_main_contract = main_contract
                        pre_close = main_contract_close
                        pre_index = index
                    else:  # 不展期
                        stay_df = df2[df2.instrument_id == pre_main_contract]
                        main_contract = stay_df.iloc[0]["instrument_id"]
                        main_contract_close = stay_df.iloc[0]["close"]
                        main_contract_volume = stay_df.iloc[0]["volume"]
                        total_volume = sum(df2["volume"])
                        main_contract_turnover = stay_df.iloc[0]["turnover"]
                        total_turnover = sum(df2["turnover"])
                        main_contract_open_interest = stay_df.iloc[0]["open_interest"]
                        total_open_interest = sum(df2["open_interest"])
                        index = pre_index * (main_contract_close / pre_close)
                        df_list.append(
                            [
                                calender_day,
                                main_contract,
                                main_contract_close,
                                main_contract_volume,
                                total_volume,
                                main_contract_turnover,
                                total_turnover,
                                main_contract_open_interest,
                                total_open_interest,
                                index,
                            ]
                        )
                        pre_close = main_contract_close
                        pre_index = index
        result = pd.DataFrame(
            df_list,
            columns=[
                "calender_day",
                "instrument_id",
                "close",
                "volume",
                "total_volume",
                "turnover",
                "total_turnover",
                "open_interest",
                "total_open_interest",
                "index",
            ],
        )

        return result


class StaticTickToK(object):
    """
    功能: 将tick数据转化为K线数据, 是静态的, 适合回测使用
    """

    def _trigger_based_resample(
        df: pd.DataFrame, time_col: str, count_limit: int, func_dict: dict
    ):
        """
        功能: 基于 "触发机制" 的重采样. 只有当第N+1次变动发生时, 才将前N次变动打包

        逻辑: 计算分钟级变动 -> 累计计数 -> 按count_limit分桶 -> 聚合

        参数:
            df: 含有datetime, last_price, volume, open_interest等列的DataFrame
            time_col: 时间列名
            count_limit: 多少个变动区间 (以1分钟为单位) 构成一个桶
            func_dict: 聚合规则字典，例如 {'volume': 'sum', 'last_price': 'last'}

        返回:
            聚合后的 DataFrame
        """
        # 1. 数据预处理 (为了不污染原数据，建议先复制，或者确保入参已是处理好的格式)
        # 这里为了函数健壮性，做一次类型转换和排序
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.sort_values(time_col, inplace=True)

        # 2. 计算分钟变动
        # 向下取整到分钟
        current_minute = df[time_col].dt.floor("min")
        # 与上一行错位比较
        shifted_minute = current_minute.shift(1)
        # 标记是否发生变化 (第一行默认为True)
        is_change = (current_minute != shifted_minute).fillna(True)

        # 3. 计算变动计数并分桶
        # 累计变动次数
        change_count = is_change.cumsum()
        # 根据count_limit进行整除分桶 (例如每3次变动归为一个桶)
        buckets = (change_count - 1) // count_limit

        # 4. 执行聚合
        # 直接利用pandas的groupby进行聚合
        result = df.groupby(buckets).agg(func_dict)

        return result

    def tick_to_K(df: pd.DataFrame, time: str) -> pd.DataFrame:
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
            df: tick行情数据, 有datetime, instrument_id, last_price, volume, open_interest等列
            time: K线周期, 如'5min', '1h', '1D'等

        返回:
            pd.DataFrame: 有datetime (日K用trading_day), instrument_id, open, high, low, close, volume, open_interest等列
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
                    vtd_list[idx]
                    if idx < len_vtd
                    else vtd_list[-1] if vtd_list else None
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
        groups = df.groupby(
            ["instrument_id", "_trade_date"], sort=False, group_keys=False
        )
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
                    r = g.resample(
                        time, on="datetime", closed="left", label="left"
                    ).agg(
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

                    # agg_rules = {
                    #     "datetime": "first",
                    #     "last_price": ["first", "max", "min", "last"],
                    #     "vol_increment": "sum",
                    #     "open_interest": "last",
                    # }
                    # r = _trigger_based_resample(
                    #     g,
                    #     time_col="datetime",
                    #     count_limit=int(time[:-3]),
                    #     func_dict=agg_rules,
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


class DynamicTickToK(object):
    """
    功能: 将tick数据转化为K线数据, 是动态的, 适合在线交易使用
    """

    def __init__(self, interval_minutes: int | None = None):
        if interval_minutes is None:
            interval_minutes = 1

        self.interval_seconds = interval_minutes * 60
        self.current_kline = None
        self.klines = []
        self.last_volume = 0.0  # 储存上一个Tick的累计成交量
        self.last_price = None  # 可选: 用于判断是否为第一个Tick

    def _create_kline(self, open_time, price: float):
        """创建一个新的K线对象"""
        kline = {
            "open_time": open_time,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
        }
        return kline

    def _update_kline(self, kline, price: float, delta_volume: int):
        """更新K线数据"""
        kline["close"] = price
        kline["high"] = max(kline["high"], price)
        kline["low"] = min(kline["low"], price)
        kline["volume"] += delta_volume

    def _finalize_kline(self, kline):
        """完成K线数据格式化"""
        return {
            "open_time": self._format_time(kline["open_time"]),
            "open": kline["open"],
            "high": kline["high"],
            "low": kline["low"],
            "close": kline["close"],
            "volume": kline["volume"],
        }

    def _format_time(self, timestamp) -> pd.Timestamp:
        """格式化时间戳"""
        return pd.Timestamp(timestamp, unit="s")

    def add_tick(self, tick_time: pd.Timestamp, price: float, cumulative_volume: int):
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
        kline_open_time = (
            int(tick_time.timestamp()) // self.interval_seconds
        ) * self.interval_seconds

        if self.current_kline is None:
            # 初始化第一根K线
            self.current_kline = self._create_kline(kline_open_time, price)
            self._update_kline(self.current_kline, price, delta_volume)
        elif kline_open_time != self.current_kline["open_time"]:
            # 当前K线结束, 保存并新建下一根
            self.klines.append(self._finalize_kline(self.current_kline))
            self.current_kline = self._create_kline(kline_open_time, price)
            self._update_kline(self.current_kline, price, delta_volume)
        else:
            # 更新当前K线
            self._update_kline(self.current_kline, price, delta_volume)

        # 更新当前K线
        self.last_price = price
        self.last_volume = cumulative_volume

    def flush(self):
        """
        功能: 输出当前正在构建的K线 (如果有的话)
        """
        if self.current_kline:
            self.klines.append(self._finalize_kline(self.current_kline))
            self.current_kline = None

    def get_klines(self):
        return self.klines

    def print_klines(self):
        for k in self.get_klines():
            print(
                f"[{k['open_time']}] O:{k['open']} H:{k['high']} L:{k['low']} C:{k['close']} V:{k['volume']}"
            )


class OtherTools(object):
    """
    功能: 封装了一些常用的工具函数
    """

    def get_maturity_date(
        instrument_id: str,
        now_date: pd.Timestamp,
        maturity_day: int | None = None,
        max_years_ahead: int | None = None,
    ) -> pd.Timestamp:
        """
        功能: 从instrument_id中提取出合约到期日

        参数:
            instrument_id: 合约代码, 例如'IF2309'
            now_date: 交易日
            maturity_day: 到期日日期 (默认28)
            max_years_ahead: 允许的最大未来年限 (默认3年)

        返回:
            pd.Timestamp: 合约到期日
        """
        if maturity_day is None:
            maturity_day = 28
        if max_years_ahead is None:
            max_years_ahead = 3

        # 1, 提取合约代码中的年份个位和月份
        match = re.search(r"(\d{3})$", instrument_id)
        if not match:
            raise ValueError(f"合约代码{instrument_id}格式错误, 未找到末尾3位数字")

        num_part = match.group(1)
        y_digit = int(num_part[-3])  # 年份个位
        month = int(num_part[-2:])  # 月份

        if not (1 <= month <= 12):
            raise ValueError(f"合约代码{instrument_id}中月份{month}无效")

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

    def cal_date_spread(
        current_date: pd.Timestamp, maturity_date: pd.Timestamp, trading_calender: list
    ) -> dict:
        """
        功能: 计算当前日期距离合约到期日还有几个月, 并计算当月已经过了多少个交易日以及还剩多少个交易日

        参数:
            current_date: 当前交易日
            maturity_date: 合约到期日
            trading_calender: 已按升序排列的交易日列表 (包含pd.Timestamp对象)

        返回:
            dict: 包含剩余月数, 当月已过/剩余交易日数等信息
        """
        # 基础校验
        if len(trading_calender) == 0:
            raise ValueError("交易日序列不能为空")

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

    def add_contract_info(codes_series: pd.Series) -> pd.DataFrame:
        """
        功能: 为入选品种(code)添加合约规模(scale), 保证金比例(margin), 手续费(fee)等合约基本数据

        参数:
            codes_series: 品种代码

        返回:
            pd.DataFrame: 含有scale, margin等列, code作为索引
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

    def update_or_add_rows(
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        keys: list,
        update_cols: list | None = None,
    ) -> pd.DataFrame:
        """
        功能: 根据指定的键 (keys), 用df_source更新df_target, 包含严格的数据结构检查

        逻辑:
            1, 检查: 验证目标表和源表的列名及数据类型是否一致
            2, 匹配到的行: 用df_source的值覆盖df_target
            3, 未匹配的行: 保留df_target的原有数据
            4, 新增的行: 将df_source中独有的行添加到df_target

        参数:
            df_target: 目标数据表
            df_source: 源数据表
            keys: 用于匹配的列名列表
            update_cols: 需要更新的列名列表, 如果为None, 则更新所有列

        返回:
            pd.DataFrame: 处理后的新数据表

        异常:
            ValueError: 当列名缺失或数据类型不匹配时抛出
        """

        def _validate_schema(
            target: pd.DataFrame,
            source: pd.DataFrame,
            operation_keys: list | None = None,
        ):
            """
            内部函数: 检查两个DataFrame的列名和类型是否一致
            """
            # 1, 检查列名是否完全一致
            cols_target = set(target.columns)
            cols_source = set(source.columns)

            if cols_target != cols_source:
                missing_in_source = cols_target - cols_source
                extra_in_source = cols_source - cols_target
                error_msg = "列名不匹配:\n"
                if missing_in_source:
                    error_msg += f"- 源表缺少列: {missing_in_source}\n"
                if extra_in_source:
                    error_msg += f"- 源表多出列: {extra_in_source}"
                raise ValueError(error_msg)

            # 2, 检查数据类型是否一致
            dtypes_target = target.dtypes
            dtypes_source = source.dtypes

            # 遍历所有列进行检查
            mismatches = []
            for col in target.columns:
                if dtypes_target[col] != dtypes_source[col]:
                    mismatches.append(
                        f"列 '{col}': 目标表为 [{dtypes_target[col]}], 源表为 [{dtypes_source[col]}]"
                    )

            if mismatches:
                error_detail = "\n".join(mismatches)
                raise ValueError(f"数据结构(类型)不一致:\n{error_detail}")

        # 1, 执行预检
        _validate_schema(df_target, df_source, keys)

        # 2, 创建副本, 防止修改原始数据
        df_t = df_target.copy()
        df_s = df_source.copy()

        # 3, 设置索引
        df_t.set_index(keys, inplace=True)
        df_s.set_index(keys, inplace=True)

        # 4, 确定要更新的列
        if update_cols is None:
            cols_to_update = df_s.columns
        else:
            cols_to_update = update_cols

        # 5, 以源表为基础, 缺失的部分用目标表填补
        result = df_s[cols_to_update].combine_first(df_t[cols_to_update])

        return result.reset_index()

    def find_per_value(
        sequence: list | pd.Series | np.ndarray, percentile_value: float
    ) -> float | None:
        """
        功能: 找出数组中处于某分位值的数

        参数:
            sequence: 数据序列
            percentile_value: 分位值 (0-1)
        """
        array = pd.Series(sequence).to_numpy()

        if not 0 <= percentile_value <= 1:
            raise ValueError("percentile_value must be between 0 and 1")

        # 获取历史值 (排除当前值)
        historical_values = array[:-1]

        # 检查历史值是否为空
        if len(historical_values) == 0:
            return None

        # 检查历史值是否全是None
        if np.all(pd.isna(historical_values)):
            return None

        # 计算分位数
        return historical_values.quantile(percentile_value)

    def find_rank_value(sequence: list | pd.Series | np.ndarray) -> float | None:
        """
        功能: 确定当前值在历史数据中的百分位

        参数:
            sequence: 数据序列
        """
        array = pd.Series(sequence).to_numpy()

        # 1. 基础长度检查
        if len(array) < 2:
            return None

        current_value = array[-1]  # 当前值
        historical_values = array[:-1]  # 前面的历史值

        # 2. 检查历史数据是否为空
        if len(historical_values) == 0:
            return None

        # 3. 如果历史数据全是None, 无法计算有意义的排名, 直接返回None
        if np.all(pd.isna(historical_values)):
            return None

        # 4. 如果当前值本身是None, 也无法计算排名
        if pd.isna(current_value):
            return None

        # 为了严谨, 只统计非None的历史值数量作为分母 (如果历史数据中有零星None)
        valid_historical_count = np.sum(~pd.isna(historical_values))

        if valid_historical_count == 0:
            return None

        count_leq_current = np.sum(
            (historical_values <= current_value) & (~pd.isna(historical_values))
        )

        return count_leq_current / valid_historical_count

    def cumulative_cal_return(df: pd.DataFrame, calc_type: str | None = None) -> list:
        """
        功能: 根据交易信号计算盈亏 (为了简化代码逻辑提高运算效率, 一次调用只能计算一个方向上的盈亏) , 支持收益率版本 ('prod') 和点差版本 ('sum')

        参数:
            df: 有datetime, last_price和trade_signal等列
            calc_type: 计算类型, 'prod'表示收益率版本, 'sum'表示点差版本
        """
        if calc_type is None:
            calc_type = "prod"

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

    def filter_trade_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        功能: 过滤向量化回测生成的无用交易信号 (为了简化代码逻辑提高运算效率, 一次调用只能过滤一个方向上的信号)

        参数:
            df: 有datetime, trade_signal等列

        返回:
            pd.DataFrame: 过滤后的交易信号
        """
        if df.empty or "trade_signal" not in df.columns:
            return df

        trade_signals = df["trade_signal"].to_numpy()

        # 找出所有值为1和0的下标
        ones_indices = np.where(trade_signals == 1)[0]
        zeros_indices = np.where(trade_signals == 0)[0]

        if len(ones_indices) == 0 or len(zeros_indices) == 0:
            # 如果没有1或0, 直接返回原DataFrame (trade_signal列全为None)
            result = df.copy()
            result["trade_signal"] = None
            return result

        # 初始化结果数组
        filtered_signals = np.full(len(trade_signals), None)

        # 使用更清晰的双指针逻辑
        one_idx, zero_idx = 0, 0
        last_valid_end = -1
        while one_idx < len(ones_indices) and zero_idx < len(zeros_indices):
            # 寻找下一个有效的1 (下标必须大于last_valid_end)
            while (
                one_idx < len(ones_indices) and ones_indices[one_idx] <= last_valid_end
            ):
                one_idx += 1

            if one_idx >= len(ones_indices):
                break

            current_one_pos = ones_indices[one_idx]

            # 寻找下一个有效的0 (下标必须大于当前1的位置)
            while (
                zero_idx < len(zeros_indices)
                and zeros_indices[zero_idx] <= current_one_pos
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

    def propagate_true_down(
        df: pd.DataFrame, col_name: str, window: int
    ) -> pd.DataFrame:
        """
        功能: df[col_name]由空值, bool值共同构成, 找到df[col_name]中为True的位置,
            将其所在行下面的window行也设为True (不管原来是不是空值)

        参数:
            df: 数据框
            col_name: 列名
            window: 向下传播的行数 (不包含当前行)

        返回:
            pd.DataFrame
        """
        df = df.copy()

        # 找出所有True的位置 (忽略None)
        true_indices = df.index[df[col_name] == True].tolist()
        for idx in true_indices:
            end_idx = idx + window
            end_idx = min(end_idx, df.index[-1])
            df.loc[idx:end_idx, col_name] = True

        return df

    def find_local_extrema(
        sequence: list | pd.Series | np.ndarray,
        window: int,
        location: int,
        extrema_type: str | None = None,
    ) -> float | dict | None:
        """
        功能: 局部极值检测函数, 可同时检测高点和低点

        参数:
            sequence: 数据序列
            window: 窗口大小
            location: 需要判断极值的位置
            extrema_type: 'high','low','both'
        """
        if extrema_type is None:
            extrema_type = "both"

        array = pd.Series(sequence).to_numpy()

        if len(array) != window:
            raise ValueError(f"输入必须是{window}长度的数据, 实际长度为{len(array)}")

        if not 1 <= location <= window:
            raise ValueError(f"location必须在1到{window}范围内, 实际值为{location}")

        target_value = array[location - 1]

        if extrema_type == "high":
            max_value = np.max(array)
            return target_value if np.isclose(target_value, max_value) else None
        elif extrema_type == "low":
            min_value = np.min(array)
            return target_value if np.isclose(target_value, min_value) else None
        elif extrema_type == "both":
            max_value = np.max(array)
            min_value = np.min(array)
            high_result = target_value if np.isclose(target_value, max_value) else None
            low_result = target_value if np.isclose(target_value, min_value) else None
            return {"high": high_result, "low": low_result}
        else:
            raise ValueError("extrema_type必须是'high','low',或'both'")

    def trend_strength_indicator(
        sequence: list | pd.Series | np.ndarray, times: int
    ) -> float:
        """
        功能: 计算趋势指标

        参数:
            sequence: 数据序列
            times: 随机选取的次数

        返回:
            float: 趋势指标
        """
        array = pd.Series(sequence).to_numpy()

        # 输入验证
        if len(array) < 2:
            raise ValueError("数组长度至少需要2个元素")

        if times <= 0:
            raise ValueError("随机选取次数必须大于0")

        # 随机选择索引对
        random_array1 = np.random.choice(len(array), size=times, replace=True)
        random_array2 = np.random.choice(len(array), size=times, replace=True)

        ratios = []
        for n1, n2 in zip(random_array1, random_array2):
            # 确保两个索引不同
            if n1 == n2:
                continue

            # 获取起始和结束索引
            start_idx = min(n1, n2)
            end_idx = max(n1, n2)

            # 避免除零错误
            if array[start_idx] == 0:
                continue

            # 计算比率变化
            ratio_change = array[end_idx] / array[start_idx] - 1
            ratios.append(ratio_change)

        # 如果没有有效数据点, 返回0
        if not ratios:
            return 0.0

        return np.mean(ratios) / np.std(ratios) if np.std(ratios) > 0 else 0.0

    def get_weight(index: str, year_month: str) -> pd.Series:
        """
        功能: 从'南华指数系列历史权重.xlsx'中获取某个指数某个日期的权重数据

        参数:
            index: 工作表名称, 也是指数的简称, 如"综合指数"
            year_month: 权重调整日期, 如"2020-06"

        参数:
            pd.Series: 权重数据
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


class KLineAnalyzer:
    """
    功能: K线分析器, 主要是观察特定时间点附近的多个时间周期的K线形态
    """

    def __init__(self):
        self.custom_style = self._create_custom_style()

    def _create_custom_style(self) -> mpf.make_mpf_style:
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
            # 备用样式同样必须包含rc, 否则会再次报错
            return mpf.make_mpf_style(
                marketcolors=mpf.make_marketcolors(up="r", down="g"),
                rc={
                    "font.family": "sans-serif",
                    "font.sans-serif": ["FZHei-B01S", "SimHei"],
                    "axes.unicode_minus": False,
                },
            )

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: list) -> bool:
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

    def _extract_by_count(
        self,
        df: pd.DataFrame,
        target_time: pd.Timestamp,
        count_before: int | None = None,
        count_after: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        """
        功能: 向前查找最接近的时间点并截取数据
        """
        if count_before is None:
            count_before = 10
        if count_after is None:
            count_after = 10

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

    def _prepare_add_plots(self, df: pd.DataFrame, panel_config: dict) -> list:
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

    def _create_plots(
        self, dfs: list, titles: list, additional_plots_configs=None
    ) -> list:
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
        dfs: list,
        instrument_id1: list,
        counts_before=None,
        counts_after=None,
        signal_column="trade_signal",
        additional_plots_configs=None,
        frequency_names=None,
    ) -> None:
        """
        参数:
            additional_plots_configs (list of dicts): 每个dict对应一个频率的附加绘图配置, 顺序必须与dfs一致.
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

            # 1. 处理高频
            window_df_hf, matched_time_hf = self._extract_by_count(
                high_freq_df, high_freq_time, counts_before[0], counts_after[0]
            )
            valid_dfs.append(window_df_hf)
            valid_titles.append(f"{instrument_id1} {frequency_names[0]}")

            # 2. 处理中/低频
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

            # 3. 绘图
            if len(valid_dfs) == len(dfs):
                self._create_plots(valid_dfs, valid_titles, additional_plots_configs)
                plt.show(block=True)
                plt.close("all")

        print("分析完成")
