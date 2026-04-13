# 该项目的作用: 测试实际投资中跟踪南华综合指数时的偏离度 保证金占用情况 (当日判定是否展期, 并在收盘时展期)
import pandas as pd
import zipfile as zp
import datetime as dt
import io, csv

import sys

sys.path.append("D:\\LearningAndWorking\\VSCode\\python\\module\\")
from package1 import Cal_index1 as ci


def get_market_data1(index_name, weight_date_list):
    """
    功能: 获取所用品种的所有日频行情数据

    参数:
        index_name(str): 指数名称
        weight_date_list(list): 回测涉及到的权重调整日期

    返回:
        df(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的数据
    """
    codes_list = []
    for weight_date in weight_date_list[:-1]:
        weight = ci.get_weight(index_name, weight_date)
        codes_list.extend(list(weight.index))

    df = pd.read_csv(
        "D:\\LearningAndWorking\\VSCode\\data\\csv\\chinese_commodity_future_trading_data(2024年及以前).csv"
    )
    result_list = []
    for code in set(codes_list):
        regex_pattern = f"^{code}[0-9]{{3,4}}$"
        mask = df.instrument_id.str.contains(regex_pattern, na=False, regex=True)
        temp_df = df.loc[mask]
        result_list.append(temp_df)

    result = pd.concat(result_list, ignore_index=True)
    result.drop(columns=["settle", "turnover", "exchange"], inplace=True)

    return result


def get_market_data2(index_name, weight_date_list, time):
    """
    功能: 获取所用品种的所有分钟级行情数据

    参数:
        index_name(str): 指数名称
        weight_date_list(list): 回测涉及到的权重调整日期
        time(str): 时间点，格式为'HH:MM:SS'

    返回:
        df(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的数据
    """
    codes_list = []
    for weight_date in weight_date_list[:-1]:
        weight = ci.get_weight(index_name, weight_date)
        codes_list.extend(list(weight.index))

    result_list = []
    with zp.ZipFile(
        "D:\\LearningAndWorking\\VSCode\\data\\csv\\全部期货合约一分钟级数据（23_12_21-24_08_28）.zip"
    ) as zf:
        for code in set(codes_list):
            content = zf.read(code + ".csv")
            text = content.decode(encoding="utf-8")
            temp_df = pd.read_csv(io.StringIO(text))
            temp_df = temp_df[temp_df.time == time]
            result_list.append(temp_df)

    result = pd.concat(result_list, ignore_index=True)
    result.drop(columns=["time", "turnover"], inplace=True)

    return result


def calculate_function1(df1, df2, df3, weights, date, path, account, fee):
    """
    功能: 计算第一个交易日的净值数据

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, code, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, margin, trading_fee, net_value等列的历史净值数据
        weights(pd.Series): 权重信息
        date(str): 日期, 格式为'YYYY-MM-DD'
        path(str): 数据输出路径
        account(float): 账户初始权益
        fee(float): 交易费用

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史净值数据
    """
    # 纳入净值计算的品种的基本信息
    df4 = ci.add_contract_info(weights)
    df4.columns = ["weight", "scale", "margin_ratio"]

    # 更新持仓信息
    list1 = []
    for code1 in df4.index:
        # 品种基本信息
        weight = df4.loc[code1, ["weight"]]  # 品种权重
        scale = df4.loc[code1, ["scale"]]  # 合约规模
        margin_ratio = df4.loc[code1, ["margin_ratio"]]  # 保证金比例

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        roll_state = 0  # 展期状态
        main_contract = df5["instrument_id"].values[0]  # 主力合约
        main_contract_price = df5["close"].values[0]  # 主力合约价格
        largest_open_interest_contract = main_contract  # 主力合约就是持仓量最大的合约
        second_contract = ""  # 次主力合约
        second_contract_price = 0  # 次主力合约价格
        open_interest1 = round(
            account * weight / (main_contract_price * scale)
        )  # 主力合约持仓
        open_interest2 = 0  # 次主力合约持仓
        margin = (
            open_interest1 * main_contract_price * scale * margin_ratio
        )  # 保证金占用金额
        trading_fee = open_interest1 * main_contract_price * scale * fee  # 交易费用
        list1.append(
            [
                date,
                code1,
                main_contract,
                main_contract_price,
                open_interest1,
                roll_state,
                second_contract,
                second_contract_price,
                open_interest2,
                largest_open_interest_contract,
                margin,
                trading_fee,
            ]
        )

    # 数据输出
    df6 = pd.DataFrame(list1)
    df6.columns = [
        "date",
        "code",
        "main_contract",
        "main_contract_price",
        "open_interest1",
        "roll_state",
        "second_contract",
        "second_contract_price",
        "open_interest2",
        "largest_open_interest_contract",
        "margin",
        "trading_fee",
    ]
    net_value = account - sum(df6["trading_fee"])  # 账户净值
    df6["net_value"] = net_value
    list2 = df6.values.tolist()
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list2)

    # 更新变量
    if df2.shape[0] > 600:
        df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
        df3 = pd.concat([df3.iloc[1:, :], df6], ignore_index=True)
    else:
        df2 = pd.concat([df2, df1], ignore_index=True)
        df3 = pd.concat([df3, df6], ignore_index=True)

    return df2, df3


def calculate_function2(df1, df2, df3, weights, date, path, fee):
    """
    功能: 计算第二, 三个交易日的净值数据

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, code, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, margin, trading_fee, net_value等列的历史净值数据
        weights(pd.Series): 权重信息
        date(str): 日期, 格式为'YYYY-MM-DD'
        path(str): 数据输出路径
        fee(float): 交易费用

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史净值数据
    """
    # 纳入净值计算的品种的基本信息
    df4 = ci.add_contract_info(weights)
    df4.columns = ["weight", "scale", "margin_ratio"]

    # 计算当日账户净值
    net_value = df3["net_value"].values[-1]
    for code1 in df4.index:
        # 前一交易日的持仓信息
        pre_main_contract = df3[df3["code"] == code1]["main_contract"].values[-1]
        pre_main_contract_price = df3[df3["code"] == code1][
            "main_contract_price"
        ].values[-1]
        pre_open_interest1 = df3[df3["code"] == code1]["open_interest1"].values[-1]

        scale = df4.loc[code1, ["scale"]]  # 合约规模

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        main_contract = pre_main_contract  # 主力合约
        main_contract_price = df5[df5["instrument_id"] == main_contract][
            "close"
        ].values[
            0
        ]  # 主力合约价格

        net_value = (
            net_value
            + pre_open_interest1
            * (main_contract_price - pre_main_contract_price)
            * scale
        )

    # 更新持仓信息
    list1 = []
    for code1 in df4.index:
        # 前一交易日的持仓信息
        pre_main_contract = df3[df3["code"] == code1]["main_contract"].values[-1]
        pre_main_contract_price = df3[df3["code"] == code1][
            "main_contract_price"
        ].values[-1]
        pre_open_interest1 = df3[df3["code"] == code1]["open_interest1"].values[-1]

        # 品种基本信息
        weight = df4.loc[code1, ["weight"]]  # 品种权重
        scale = df4.loc[code1, ["scale"]]  # 合约规模
        margin_ratio = df4.loc[code1, ["margin_ratio"]]  # 保证金比例

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        roll_state = 0  # 展期状态
        main_contract = pre_main_contract  # 主力合约
        main_contract_price = df5[df5["instrument_id"] == main_contract][
            "close"
        ].values[
            0
        ]  # 主力合约价格
        largest_open_interest_contract = df5["instrument_id"].values[
            0
        ]  # 持仓量最大的合约
        second_contract = ""  # 次主力合约
        second_contract_price = 0  # 次主力合约价格
        open_interest1 = round(
            net_value * weight / (main_contract_price * scale)
        )  # 主力合约持仓
        open_interest2 = 0  # 次主力合约持仓
        margin = (
            open_interest1 * main_contract_price * scale * margin_ratio
        )  # 保证金占用金额
        trading_fee = (
            abs(open_interest1 - pre_open_interest1) * main_contract_price * scale * fee
        )  # 交易费用
        list1.append(
            [
                date,
                code1,
                main_contract,
                main_contract_price,
                open_interest1,
                roll_state,
                second_contract,
                second_contract_price,
                open_interest2,
                largest_open_interest_contract,
                margin,
                trading_fee,
            ]
        )

    # 数据输出
    df6 = pd.DataFrame(list1)
    df6.columns = [
        "date",
        "code",
        "main_contract",
        "main_contract_price",
        "open_interest1",
        "roll_state",
        "second_contract",
        "second_contract_price",
        "open_interest2",
        "largest_open_interest_contract",
        "margin",
        "trading_fee",
    ]
    net_value = net_value - sum(df6["trading_fee"])  # 账户净值
    df6["net_value"] = net_value
    list2 = df6.values.tolist()
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list2)

    # 更新变量
    if df2.shape[0] > 600:
        df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
        df3 = pd.concat([df3.iloc[1:, :], df6], ignore_index=True)
    else:
        df2 = pd.concat([df2, df1], ignore_index=True)
        df3 = pd.concat([df3, df6], ignore_index=True)

    return df2, df3


def calculate_weight_constant_function(df, df1, df2, df3, weights, date, path, fee):
    """
    功能: 计算非权重调整日的净值数据

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, code, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, margin, trading_fee, net_value等列的历史净值数据
        weights(pd.Series): 权重信息
        date(str): 日期, 格式为'YYYY-MM-DD'
        path(str): 数据输出路径
        fee(float): 交易费用

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史净值数据
    """
    # 纳入指数计算的品种的基本信息
    df4 = ci.add_contract_info(weights)
    df4.columns = ["weight", "scale", "margin_ratio"]

    # 计算当日账户净值
    net_value = df3["net_value"].values[-1]
    for code1 in df4.index:
        # 前一个交易日的持仓信息
        pre_roll_state = df3[df3["code"] == code1]["roll_state"].values[-1]
        pre_main_contract = df3[df3["code"] == code1]["main_contract"].values[-1]
        pre_main_contract_price = df3[df3["code"] == code1][
            "main_contract_price"
        ].values[-1]
        pre_second_contract = df3[df3["code"] == code1]["second_contract"].values[-1]
        pre_second_contract_price = df3[df3["code"] == code1][
            "second_contract_price"
        ].values[-1]
        pre_open_interest1 = df3[df3["code"] == code1]["open_interest1"].values[-1]
        pre_open_interest2 = df3[df3["code"] == code1]["open_interest2"].values[-1]

        scale = df4.loc[code1, ["scale"]]  # 合约规模

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        if pre_roll_state == 0:
            net_value = (
                net_value
                + pre_open_interest1
                * (
                    df5[df5["instrument_id"] == pre_main_contract]["close"].values[0]
                    - pre_main_contract_price
                )
                * scale
            )
        else:
            net_value = (
                net_value
                + (
                    pre_open_interest1
                    * (
                        df5[df5["instrument_id"] == pre_main_contract]["close"].values[
                            0
                        ]
                        - pre_main_contract_price
                    )
                    + pre_open_interest2
                    * (
                        df5[df5["instrument_id"] == pre_second_contract][
                            "close"
                        ].values[0]
                        - pre_second_contract_price
                    )
                )
                * scale
            )

    # 更新持仓信息
    list1 = []
    for code1 in df4.index:
        # 前一个交易日的持仓信息
        pre_roll_state = df3[df3["code"] == code1]["roll_state"].values[-1]
        pre_main_contract = df3[df3["code"] == code1]["main_contract"].values[-1]
        pre_main_contract_price = df3[df3["code"] == code1][
            "main_contract_price"
        ].values[-1]
        pre_largest_open_interest_contract = df3[df3["code"] == code1][
            "largest_open_interest_contract"
        ].values[-1]
        pre_second_contract = df3[df3["code"] == code1]["second_contract"].values[-1]
        pre_second_contract_price = df3[df3["code"] == code1][
            "second_contract_price"
        ].values[-1]
        pre_open_interest1 = df3[df3["code"] == code1]["open_interest1"].values[-1]
        pre_open_interest2 = df3[df3["code"] == code1]["open_interest2"].values[-1]

        # 品种基本信息
        weight = df4.loc[code1, ["weight"]]  # 品种权重
        scale = df4.loc[code1, ["scale"]]  # 合约规模
        margin_ratio = df4.loc[code1, ["margin_ratio"]]  # 保证金比例

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        if pre_roll_state == 0:
            # 当日持仓信息
            main_contract = pre_main_contract  # 主力合约
            main_contract_price = df5[df5["instrument_id"] == pre_main_contract][
                "close"
            ].values[
                0
            ]  # 主力合约价格
            largest_open_interest_contract = df5["instrument_id"].values[
                0
            ]  # 持仓量最大的合约
            roll_condition = ci.cal_date_spread(
                pd.Timestamp(date),
                ci.get_maturity_date(main_contract, pd.Timestamp(date)),
                pd.to_datetime(sorted(set(df["calender_day"]))),
            )
            if (roll_condition["months_remaining"] <= 2) & (
                roll_condition["trading_days_left_this_month"] <= 5
            ):
                roll_df = df5[
                    df5["instrument_id"].apply(
                        lambda x: (
                            True
                            if ci.get_maturity_date(x, pd.Timestamp(date))
                            > ci.get_maturity_date(main_contract, pd.Timestamp(date))
                            else False
                        )
                    )
                ]  # 筛选出比当前主力合约更晚到期的合约
                roll_state = 1  # 展期状态
                second_contract = roll_df["instrument_id"].values[0]
                second_contract_price = roll_df["close"].values[0]
                open_interest1 = round(
                    (net_value * 0.8 * weight) / (main_contract_price * scale)
                )  # 主力合约持仓
                open_interest2 = round(
                    (net_value * 0.2 * weight) / (second_contract_price * scale)
                )
                margin = (
                    (
                        open_interest1 * main_contract_price
                        + open_interest2 * second_contract_price
                    )
                    * scale
                    * margin_ratio
                )  # 保证金占用金额
                trading_fee = (
                    (
                        abs(open_interest1 - pre_open_interest1) * main_contract_price
                        + abs(open_interest2 - pre_open_interest2)
                        * second_contract_price
                    )
                    * scale
                    * fee
                )  # 交易费用
            else:
                if len(
                    set(
                        df3[df3["code"] == code1].tail(3)[
                            "largest_open_interest_contract"
                        ]
                    )
                ) == 1 & (
                    ci.get_maturity_date(
                        largest_open_interest_contract, pd.Timestamp(date)
                    )
                    > ci.get_maturity_date(main_contract, pd.Timestamp(date))
                ):  # 如果最大持仓量的合约连续三天是同一合约且最大持仓量的合约比主力合约更晚到期
                    roll_state = 1  # 展期状态
                    second_contract = pre_largest_open_interest_contract
                    second_contract_price = df5[
                        df5["instrument_id"] == second_contract
                    ]["close"].values[0]
                    open_interest1 = round(
                        (net_value * 0.8 * weight) / (main_contract_price * scale)
                    )  # 主力合约持仓
                    open_interest2 = round(
                        (net_value * 0.2 * weight) / (second_contract_price * scale)
                    )
                    margin = (
                        (
                            open_interest1 * main_contract_price
                            + open_interest2 * second_contract_price
                        )
                        * scale
                        * margin_ratio
                    )  # 保证金占用金额
                    trading_fee = (
                        (
                            abs(open_interest1 - pre_open_interest1)
                            * main_contract_price
                            + abs(open_interest2 - pre_open_interest2)
                            * second_contract_price
                        )
                        * scale
                        * fee
                    )  # 交易费用
                else:  # 不展期
                    roll_state = 0
                    second_contract = ""  # 次主力合约
                    second_contract_price = 0  # 次主力合约价格
                    open_interest1 = round(
                        (net_value * weight) / (main_contract_price * scale)
                    )  # 主力合约持仓
                    open_interest2 = 0  # 次主力合约持仓
                    margin = (
                        open_interest1 * main_contract_price * scale * margin_ratio
                    )  # 保证金占用金额
                    trading_fee = (
                        abs(open_interest1 - pre_open_interest1)
                        * main_contract_price
                        * scale
                        * fee
                    )  # 交易费用
            list1.append(
                [
                    date,
                    code1,
                    main_contract,
                    main_contract_price,
                    open_interest1,
                    roll_state,
                    second_contract,
                    second_contract_price,
                    open_interest2,
                    largest_open_interest_contract,
                    margin,
                    trading_fee,
                ]
            )
        elif pre_roll_state == 1:
            main_contract = pre_main_contract
            main_contract_price = df5[df5["instrument_id"] == main_contract][
                "close"
            ].values[
                0
            ]  # 主力合约价格
            largest_open_interest_contract = df5["instrument_id"].values[
                0
            ]  # 持仓量最大的合约
            roll_state = 2  # 展期状态
            second_contract = pre_second_contract
            second_contract_price = df5[df5["instrument_id"] == second_contract][
                "close"
            ].values[0]
            open_interest1 = round(
                (net_value * 0.6 * weight) / (main_contract_price * scale)
            )  # 主力合约持仓
            open_interest2 = round(
                (net_value * 0.4 * weight) / (second_contract_price * scale)
            )
            margin = (
                (
                    open_interest1 * main_contract_price
                    + open_interest2 * second_contract_price
                )
                * scale
                * margin_ratio
            )  # 保证金占用金额
            trading_fee = (
                (
                    abs(open_interest1 - pre_open_interest1) * main_contract_price
                    + abs(open_interest2 - pre_open_interest2) * second_contract_price
                )
                * scale
                * fee
            )  # 交易费用
            list1.append(
                [
                    date,
                    code1,
                    main_contract,
                    main_contract_price,
                    open_interest1,
                    roll_state,
                    second_contract,
                    second_contract_price,
                    open_interest2,
                    largest_open_interest_contract,
                    margin,
                    trading_fee,
                ]
            )
        elif pre_roll_state == 2:
            main_contract = pre_main_contract
            main_contract_price = df5[df5["instrument_id"] == main_contract][
                "close"
            ].values[
                0
            ]  # 主力合约价格
            largest_open_interest_contract = df5["instrument_id"].values[
                0
            ]  # 持仓量最大的合约
            roll_state = 3  # 展期状态
            second_contract = pre_second_contract
            second_contract_price = df5[df5["instrument_id"] == second_contract][
                "close"
            ].values[0]
            open_interest1 = round(
                (net_value * 0.4 * weight) / (main_contract_price * scale)
            )  # 主力合约持仓
            open_interest2 = round(
                (net_value * 0.6 * weight) / (second_contract_price * scale)
            )
            margin = (
                (
                    open_interest1 * main_contract_price
                    + open_interest2 * second_contract_price
                )
                * scale
                * margin_ratio
            )  # 保证金占用金额
            trading_fee = (
                (
                    abs(open_interest1 - pre_open_interest1) * main_contract_price
                    + abs(open_interest2 - pre_open_interest2) * second_contract_price
                )
                * scale
                * fee
            )  # 交易费用
            list1.append(
                [
                    date,
                    code1,
                    main_contract,
                    main_contract_price,
                    open_interest1,
                    roll_state,
                    second_contract,
                    second_contract_price,
                    open_interest2,
                    largest_open_interest_contract,
                    margin,
                    trading_fee,
                ]
            )
        elif pre_roll_state == 3:
            main_contract = pre_main_contract
            main_contract_price = df5[df5["instrument_id"] == main_contract][
                "close"
            ].values[
                0
            ]  # 主力合约价格
            largest_open_interest_contract = df5["instrument_id"].values[
                0
            ]  # 持仓量最大的合约
            roll_state = 4  # 展期状态
            second_contract = pre_second_contract
            second_contract_price = df5[df5["instrument_id"] == second_contract][
                "close"
            ].values[0]
            open_interest1 = round(
                (net_value * 0.8 * weight) / (main_contract_price * scale)
            )  # 主力合约持仓
            open_interest2 = round(
                (net_value * 0.2 * weight) / (second_contract_price * scale)
            )
            margin = (
                (
                    open_interest1 * main_contract_price
                    + open_interest2 * second_contract_price
                )
                * scale
                * margin_ratio
            )  # 保证金占用金额
            trading_fee = (
                (
                    abs(open_interest1 - pre_open_interest1) * main_contract_price
                    + abs(open_interest2 - pre_open_interest2) * second_contract_price
                )
                * scale
                * fee
            )  # 交易费用
            list1.append(
                [
                    date,
                    code1,
                    main_contract,
                    main_contract_price,
                    open_interest1,
                    roll_state,
                    second_contract,
                    second_contract_price,
                    open_interest2,
                    largest_open_interest_contract,
                    margin,
                    trading_fee,
                ]
            )
        elif pre_roll_state == 4:
            main_contract = pre_second_contract
            main_contract_price = df5[df5["instrument_id"] == main_contract][
                "close"
            ].values[
                0
            ]  # 主力合约价格
            largest_open_interest_contract = df5["instrument_id"].values[
                0
            ]  # 持仓量最大的合约
            roll_state = 0  # 展期状态
            second_contract = ""
            second_contract_price = 0
            open_interest1 = round(
                (net_value * weight) / (main_contract_price * scale)
            )  # 主力合约持仓
            open_interest2 = 0
            margin = (
                open_interest1 * main_contract_price * scale * margin_ratio
            )  # 保证金占用金额
            trading_fee = (
                (
                    abs(open_interest1 - pre_open_interest2) * main_contract_price
                    + abs(0 - pre_open_interest1) * pre_main_contract_price
                )
                * scale
                * fee
            )  # 交易费用
            list1.append(
                [
                    date,
                    code1,
                    main_contract,
                    main_contract_price,
                    open_interest1,
                    roll_state,
                    second_contract,
                    second_contract_price,
                    open_interest2,
                    largest_open_interest_contract,
                    margin,
                    trading_fee,
                ]
            )

    # 数据输出
    df6 = pd.DataFrame(list1)
    df6.columns = [
        "date",
        "code",
        "main_contract",
        "main_contract_price",
        "open_interest1",
        "roll_state",
        "second_contract",
        "second_contract_price",
        "open_interest2",
        "largest_open_interest_contract",
        "margin",
        "trading_fee",
    ]
    net_value = net_value - sum(df6["trading_fee"])  # 账户净值
    df6["net_value"] = net_value
    list2 = df6.values.tolist()
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list2)

    # 更新变量
    if df2.shape[0] > 600:
        df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
        df3 = pd.concat([df3.iloc[1:, :], df6], ignore_index=True)
    else:
        df2 = pd.concat([df2, df1], ignore_index=True)
        df3 = pd.concat([df3, df6], ignore_index=True)

    return df2, df3


def calculate_weight_change_function(df, df1, df2, df3, weights, date, path, fee):
    """
    功能: 计算权重调整日的净值数据

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, code, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, margin, trading_fee, net_value等列的历史净值数据
        weights(pd.Series): 权重信息
        date(str): 日期, 格式为'YYYY-MM-DD'
        path(str): 数据输出路径
        fee(float): 交易费用

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史净值数据
    """
    # 纳入净值计算的品种的基本信息
    df4 = ci.add_contract_info(weights)
    df4.columns = ["weight", "scale", "margin_ratio"]

    # 计算当日账户净值
    net_value = df3["net_value"].values[-1]
    for code1 in df3[
        df3["date"] == date
    ].code:  # 因为权重调整日的历史持仓品种和将要持仓品种可能不同, 此处与其它函数计算当日账户净值的代码略有不同
        # 前一个交易日的持仓信息
        pre_roll_state = df3[df3["code"] == code1]["roll_state"].values[-1]
        pre_main_contract = df3[df3["code"] == code1]["main_contract"].values[-1]
        pre_main_contract_price = df3[df3["code"] == code1][
            "main_contract_price"
        ].values[-1]
        pre_second_contract = df3[df3["code"] == code1]["second_contract"].values[-1]
        pre_second_contract_price = df3[df3["code"] == code1][
            "second_contract_price"
        ].values[-1]
        pre_open_interest1 = df3[df3["code"] == code1]["open_interest1"].values[-1]
        pre_open_interest2 = df3[df3["code"] == code1]["open_interest2"].values[-1]

        scale = df4.loc[code1, ["scale"]]  # 合约规模

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        if pre_roll_state == 0:
            net_value = (
                net_value
                + pre_open_interest1
                * (
                    df5[df5["instrument_id"] == pre_main_contract]["close"].values[0]
                    - pre_main_contract_price
                )
                * scale
            )
        else:
            net_value = (
                net_value
                + (
                    pre_open_interest1
                    * (
                        df5[df5["instrument_id"] == pre_main_contract]["close"].values[
                            0
                        ]
                        - pre_main_contract_price
                    )
                    + pre_open_interest2
                    * (
                        df5[df5["instrument_id"] == pre_second_contract][
                            "close"
                        ].values[0]
                        - pre_second_contract_price
                    )
                )
                * scale
            )

    # 更新持仓信息
    list1 = []
    for code1 in df4.index:
        # 品种基本信息
        weight = df4.loc[code1, ["weight"]]  # 品种权重
        scale = df4.loc[code1, ["scale"]]  # 合约规模
        margin_ratio = df4.loc[code1, ["margin_ratio"]]  # 保证金比例

        # 对应品种的行情信息
        df5 = df1.query(
            "instrument_id.str.contains('^" + code1 + "[0-9]{3,4}', regex=True)",
            engine="python",
        )
        df5 = df5.sort_values(by="open_interest", ascending=False)  # 按持仓量进行排序

        # 当日持仓信息
        main_contract = df5["instrument_id"].values[0]  # 主力合约
        main_contract_price = df5["close"].values[0]  # 主力合约价格
        largest_open_interest_contract = main_contract  # 主力合约就是持仓量最大的合约
        roll_state = 0  # 展期状态
        second_contract = ""  # 次主力合约
        second_contract_price = 0  # 次主力合约价格
        open_interest1 = round(
            net_value * weight / (main_contract_price * scale)
        )  # 主力合约持仓
        open_interest2 = 0  # 次主力合约持仓
        margin = (
            open_interest1 * main_contract_price * scale * margin_ratio
        )  # 保证金占用金额
        trading_fee = open_interest1 * main_contract_price * scale * fee  # 交易费用
        list1.append(
            [
                date,
                code1,
                main_contract,
                main_contract_price,
                open_interest1,
                roll_state,
                second_contract,
                second_contract_price,
                open_interest2,
                largest_open_interest_contract,
                margin,
                trading_fee,
            ]
        )

    # 数据输出
    df6 = pd.DataFrame(list1)
    df6.columns = [
        "date",
        "code",
        "main_contract",
        "main_contract_price",
        "open_interest1",
        "roll_state",
        "second_contract",
        "second_contract_price",
        "open_interest2",
        "largest_open_interest_contract",
        "margin",
        "trading_fee",
    ]
    net_value = net_value - sum(df6["trading_fee"])  # 账户净值
    df6["net_value"] = net_value
    list2 = df6.values.tolist()
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(list2)

    # 更新变量
    if df2.shape[0] > 600:
        df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
        df3 = pd.concat([df3.iloc[1:, :], df6], ignore_index=True)
    else:
        df2 = pd.concat([df2, df1], ignore_index=True)
        df3 = pd.concat([df3, df6], ignore_index=True)

    return df2, df3


def main(index_name, weight_date_list, path, account, fee):
    """
    功能: 主函数, 执行回测

    参数:
        index_name(str): 指数名称
        weight_date_list(list): 历史权重调整的日期列表, 格式为'YYYY-mm'
        account(float): 账户初始净值
        path(str): 数据输出路径
        fee(float): 交易费率

    返回:
        None
    """
    df = get_market_data1(index_name, weight_date_list)  # 获取所用合约的所有行情数据
    date_list1 = sorted(
        set(df[df.calender_day >= weight_date_list[0]]["calender_day"])
    )  # 生成一个回测用的日期流

    # 初始化部分参数并计算前三个交易日的账户净值
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    weight1 = ci.get_weight(index_name, weight_date_list[0])
    df2, df3 = calculate_function1(
        df[df.calender_day == date_list1[0]],
        df2,
        df3,
        weight1,
        date_list1[0],
        path,
        account,
        fee,
    )
    df2, df3 = calculate_function2(
        df[df.calender_day == date_list1[1]],
        df2,
        df3,
        weight1,
        date_list1[1],
        path,
        fee,
    )
    df2, df3 = calculate_function2(
        df[df.calender_day == date_list1[2]],
        df2,
        df3,
        weight1,
        date_list1[2],
        path,
        fee,
    )

    n1 = 1
    for date1 in date_list1[3:]:
        df1 = df[df.calender_day == date1]
        if n1 < len(weight_date_list) and dt.datetime.strptime(
            date1, "%Y-%m-%d"
        ) >= dt.datetime.strptime(
            weight_date_list[n1], "%Y-%m"
        ):  # 判断是否需要调整权重
            weight1 = ci.get_weight(index_name, weight_date_list[n1])  # 更新权重信息
            df2, df3 = calculate_weight_change_function(
                df, df1, df2, df3, weight1, date1, path, fee
            )
            n1 = n1 + 1
        else:
            df2, df3 = calculate_weight_constant_function(
                df, df1, df2, df3, weight1, date1, path, fee
            )


if __name__ == "__main__":
    index_name1 = "综合指数"  # 指数名称
    # 历史权重调整的日期, 格式为'YYYY-mm', 最后还要补充一个下一次调整权重的日期
    weight_date_list1 = [
        "2004-06",
        "2005-06",
        "2006-06",
        "2007-06",
        "2008-06",
        "2009-06",
        "2010-06",
        "2011-06",
        "2012-06",
        "2012-09",
        "2013-06",
        "2014-06",
        "2015-03",
        "2015-06",
        "2016-06",
        "2017-06",
        "2018-06",
        "2019-06",
        "2020-06",
        "2021-06",
        "2022-06",
        "2022-09",
        "2023-06",
        "2024-06",
        "2025-06",
    ]
    path1 = f"D:\\LearningAndWorking\\VSCode\\python\\project1\\{index_name1}.csv"  # 数据输出路径
    account1 = 10000000  # 账户初始净值
    fee1 = 0.0005  # 交易费率

    list1 = [
        "date",
        "code",
        "main_contract",
        "main_contract_price",
        "open_interest1",
        "roll_state",
        "second_contract",
        "second_contract_price",
        "open_interest2",
        "largest_open_interest_contract",
        "margin",
        "trading_fee",
        "net_value",
    ]
    with open(path1, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list1)

    main(index_name1, weight_date_list1, path1, account1, fee1)
