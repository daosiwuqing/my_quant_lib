# 该项目的作用：根据一些资产配置模型确定配置权重
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def generate_weight1(
    low, high, n
):  # 随机生成每个权重都在low和high之间，且整个随机序列之和为1的随机权重序列，n为序列数
    list1 = [1]
    while min(list1) < 0.02 or max(list1) > 0.25:
        list1.clear()
        array1 = np.random.randint(
            low, high, n
        )  # rand函数生成服从均匀分布的随机数或随机数数组；randn函数生成服从正态分布的随机数或随机数数组；randint函数可以生成给定上下限范围的随机数
        list1 = list(array1 / np.sum(array1))

    return list1


def generate_weight2(
    df1, low, high, n
):  # 根据马科维茨理论生成权重，df1是dataframe格式的时间序列数据，日期是索引，每个权重都在low和high之间，n是权重序列数
    log_return = np.log(df1 / df1.shift(1)).dropna()  # 计算对数收益率（日）
    annual_return = np.exp(log_return.mean() * 245) - 1  # 年化收益率

    # 求解马科维茨理论下的最优权重
    # 初始值：
    initial_weight = np.ones(n) / n

    # 限制条件：
    def constraint1(x):
        return np.sum(x) - 1

    cons = {"type": "eq", "fun": constraint1}
    # 参数的上下限：
    bound = tuple((low, high) for x in initial_weight)
    # 目标函数：
    fun = lambda x: np.sqrt(x.T.dot((log_return.cov() * 245).dot(x))) / np.sum(
        x * annual_return
    )  # np.sum(x*annual_return)是组合期望收益，np.sqrt(x.T.dot((log_return.cov()*245).dot(x)))是组合协方差矩阵
    # disp打印出收敛信息，maxiter迭代的最大次数，ftol迭代终止的允许误差
    options = {"disp": False, "maxiter": 1000, "ftol": 1e-20}
    # 求解：
    res = minimize(
        fun,
        initial_weight,
        method="SLSQP",
        bounds=bound,
        constraints=cons,
        options=options,
    )
    # print(res.fun)
    # print(res.success)
    # print(res.x)#打印求解结果
    weight = list(zip(df1.columns, res.x))
    weight = pd.DataFrame(weight, columns=["code", "weight"])

    return weight


def generate_weight3(
    df1, low, high, n
):  # 根据风险平价理论生成权重，df1是dataframe格式的时间序列数据，日期是索引，每个权重都在low和high之间，n是权重序列数
    log_return = np.log(df1 / df1.shift(1)).dropna()  # 计算对数收益率（日）
    return_cov = np.array(log_return.cov())  # 计算协方差

    # 求解风险平价理论下的最优权重
    # 初始值：
    initial_weight = np.ones(n) / n

    # 限制条件：
    def constraint1(x):
        return np.sum(x) - 1

    cons = {"type": "eq", "fun": constraint1}
    # 参数的上下限：
    bound = tuple((low, high) for x in initial_weight)

    # 目标函数：
    def risk_budget_objective(weight, cov):  # weights一维权重数组，cov协方差数组
        sigma = np.sqrt(np.dot(weight, np.dot(cov, weight)))  # 计算组合标准差
        MRC = np.dot(cov, weight) / sigma
        TRC = weight * MRC
        delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
        return sum(delta_TRC)

    # disp打印出收敛信息，maxiter迭代的最大次数，ftol迭代终止的允许误差
    options = {"disp": False, "maxiter": 1000, "ftol": 1e-20}
    # 求解：
    res = minimize(
        risk_budget_objective,
        initial_weight,
        args=(return_cov),
        method="SLSQP",
        bounds=bound,
        constraints=cons,
        options=options,
    )

    weight = list(zip(df1.columns, res.x))
    weight = pd.DataFrame(weight, columns=["code", "weight"])

    return weight


if __name__ == "__main__":
    df1 = pd.read_csv(
        "D:\\LearningAndWorking\\VSCode\\python\\project4\\INDEX.csv", index_col="date"
    )
    result = generate_weight2(df1, -1, 1, 3)
    print(result)
