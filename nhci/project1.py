# 该项目的作用: 构建南华单品种指数 (当日判定是否展期, 并在收盘时展期)
import pandas as pd
import zipfile as zp
import io, csv

import sys

sys.path.append("D:\\LearningAndWorking\\VSCode\\python\\module\\")
from package1 import Cal_index1 as ci


def get_market_data1(code):
    """
    功能: 获取指定品种的日频行情数据

    参数:
        code(str): 期货合约代码

    返回:
        df(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的日频行情数据
    """
    df = pd.read_csv(
        "D:\\LearningAndWorking\\VSCode\\data\\csv\\chinese_commodity_future_trading_data(2024年及以前).csv"
    )
    regex_pattern = f"^{code}[0-9]{{3,4}}$"
    mask = df.instrument_id.str.contains(regex_pattern, na=False)
    df = df.loc[mask]
    df.drop(columns=["settle", "turnover", "exchange"], inplace=True)

    return df


def get_market_data2(code, time):
    """
    功能: 获取指定品种在指定时间点的分钟级行情数据

    参数:
        code(str): 期货合约代码
        time(str): 时间点, 格式为'15:00:00'

    返回:
        df(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的分钟级行情数据
    """
    with zp.ZipFile(
        "D:\\LearningAndWorking\\VSCode\\data\\csv\\全部期货合约一分钟级数据（23_12_21-24_08_28）.zip"
    ) as zf:
        content = zf.read(code + ".csv")
        text = content.decode(encoding="utf-8")
        df = pd.read_csv(io.StringIO(text))
        df = df[df.time == time]
        df = df.drop(columns=["time", "turnover"])
        df = df.rename(columns={"date": "calender_day", "code": "instrument_id"})

    return df


def roll_calculate_function(df1, df2, df3, date, path):
    """
    功能: 进行展期运算

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, index等列的历史指数数据
        date(str): 日期
        path(str): 数据输出路径

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史指数数据
    """
    # 1, 进行展期
    if df3["roll_state"].values[-1] == 0:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 1
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        roll_df = df1[
            df1["instrument_id"].apply(
                lambda x: (
                    True
                    if ci.get_maturity_date(x, pd.Timestamp(date))
                    > ci.get_maturity_date(main_contract, pd.Timestamp(date))
                    else False
                )
            )
        ]  # 筛选出比当前主力合约更晚到期的合约
        second_contract = roll_df["instrument_id"].values[0]
        second_contract_price = roll_df[roll_df["instrument_id"] == second_contract][
            "close"
        ].values[0]
        index = pre_index * (main_contract_price / pre_main_contract_price)
    elif df3["roll_state"].values[-1] == 1:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_second_contract = df3["second_contract"].values[-1]
        pre_second_contract_price = df3["second_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 2
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = pre_second_contract
        second_contract_price = df1[df1["instrument_id"] == second_contract][
            "close"
        ].values[0]
        index = pre_index * (
            (main_contract_price / pre_main_contract_price) * 0.8
            + (second_contract_price / pre_second_contract_price) * 0.2
        )
    elif df3["roll_state"].values[-1] == 2:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_second_contract = df3["second_contract"].values[-1]
        pre_second_contract_price = df3["second_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 3
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = pre_second_contract
        second_contract_price = df1[df1["instrument_id"] == second_contract][
            "close"
        ].values[0]
        index = pre_index * (
            (main_contract_price / pre_main_contract_price) * 0.6
            + (second_contract_price / pre_second_contract_price) * 0.4
        )
    elif df3["roll_state"].values[-1] == 3:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_second_contract = df3["second_contract"].values[-1]
        pre_second_contract_price = df3["second_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 4
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = pre_second_contract
        second_contract_price = df1[df1["instrument_id"] == second_contract][
            "close"
        ].values[0]
        index = pre_index * (
            (main_contract_price / pre_main_contract_price) * 0.4
            + (second_contract_price / pre_second_contract_price) * 0.6
        )
    elif df3["roll_state"].values[-1] == 4:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_second_contract = df3["second_contract"].values[-1]
        pre_second_contract_price = df3["second_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 5
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = pre_second_contract
        second_contract_price = df1[df1["instrument_id"] == second_contract][
            "close"
        ].values[0]
        index = pre_index * (
            (main_contract_price / pre_main_contract_price) * 0.2
            + (second_contract_price / pre_second_contract_price) * 0.8
        )
    elif df3["roll_state"].values[-1] == 5:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_second_contract = df3["second_contract"].values[-1]
        pre_second_contract_price = df3["second_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 0
        main_contract = pre_second_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = ""
        second_contract_price = 0
        index = pre_index * (main_contract_price / pre_second_contract_price)

    # 2, 输出数据
    list1 = [
        date,
        main_contract,
        main_contract_price,
        roll_state,
        second_contract,
        second_contract_price,
        largest_open_interest_contract,
        index,
    ]
    with open(path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list1)

    # 3, 更新变量
    new_row = pd.DataFrame(
        [list1],
        columns=[
            "date",
            "main_contract",
            "main_contract_price",
            "roll_state",
            "second_contract",
            "second_contract_price",
            "largest_open_interest_contract",
            "index",
        ],
    )
    if df2.shape[0] > 30:
        df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
        df3 = pd.concat([df3.iloc[1:, :], new_row], ignore_index=True)
    else:
        if df2.empty:
            df2 = df1.copy()
        else:
            df2 = pd.concat([df2, df1], ignore_index=True)
        if df3.empty:
            df3 = new_row.copy()
        else:
            df3 = pd.concat([df3, new_row], ignore_index=True)

    return df2, df3


def normal_calculate_function(df1, df2, df3, date, path):
    """
    功能: 进行正常计算

    参数:
        df1(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的当前行情数据
        df2(pd.DataFrame): 含有calender_day, instrument_id, close, volume, open_interest等列的历史行情数据
        df3(pd.DataFrame): 含有date, main_contract, main_contract_price, roll_state, second_contract, second_contract_price, largest_open_interest_contract, index等列的历史指数数据
        date(str): 日期
        path(str): 数据输出路径

    返回:
        df2(pd.DataFrame): 更新后的历史行情数据
        df3(pd.DataFrame): 更新后的历史指数数据
    """
    if df3["roll_state"].values[-1] > 0:  # 开始展期
        df2, df3 = roll_calculate_function(df1, df2, df3, date, path)
    else:
        pre_main_contract = df3["main_contract"].values[-1]
        pre_main_contract_price = df3["main_contract_price"].values[-1]
        pre_index = df3["index"].values[-1]

        roll_state = 0
        main_contract = pre_main_contract
        main_contract_price = df1[df1["instrument_id"] == main_contract][
            "close"
        ].values[0]
        largest_open_interest_contract = df1["instrument_id"].values[0]
        second_contract = ""
        second_contract_price = 0
        index = pre_index * (main_contract_price / pre_main_contract_price)

        list1 = [
            date,
            main_contract,
            main_contract_price,
            roll_state,
            second_contract,
            second_contract_price,
            largest_open_interest_contract,
            index,
        ]
        with open(path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list1)

        new_row = pd.DataFrame(
            [list1],
            columns=[
                "date",
                "main_contract",
                "main_contract_price",
                "roll_state",
                "second_contract",
                "second_contract_price",
                "largest_open_interest_contract",
                "index",
            ],
        )
        if df2.shape[0] > 30:
            df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
            df3 = pd.concat([df3.iloc[1:, :], new_row], ignore_index=True)
        else:
            if df2.empty:
                df2 = df1.copy()
            else:
                df2 = pd.concat([df2, df1], ignore_index=True)
            if df3.empty:
                df3 = new_row.copy()
            else:
                df3 = pd.concat([df3, new_row], ignore_index=True)

    return df2, df3


def main(code, path):
    """
    功能: 主函数

    参数:
        code(str): 期货品种代码
        path(str): 数据输出路径

    返回:
        None
    """
    df = get_market_data1(code)
    # df = get_market_data2(code, time="15:00:00")

    df2 = pd.DataFrame(
        columns=[
            "date",
            "contract",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    )
    df3 = pd.DataFrame(
        columns=[
            "date",
            "main_contract",
            "main_contract_price",
            "roll_state",
            "second_contract",
            "second_contract_price",
            "largest_open_interest_contract",
            "index",
        ]
    )
    for date1 in sorted(set(df.calender_day)):
        df1 = df[df.calender_day == date1].sort_values(
            by="open_interest", ascending=False
        )  # 对date1的合约进行持仓量排序
        if df3.shape[0] == 0:
            roll_state = 0
            main_contract = df1["instrument_id"].values[0]
            main_contract_price = df1["close"].values[0]
            largest_open_interest_contract = df1["instrument_id"].values[0]
            second_contract = ""
            second_contract_price = 0
            index = 1000

            list1 = [
                date1,
                main_contract,
                main_contract_price,
                roll_state,
                second_contract,
                second_contract_price,
                largest_open_interest_contract,
                index,
            ]
            with open(path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list1)

            new_row = pd.DataFrame(
                [list1],
                columns=[
                    "date",
                    "main_contract",
                    "main_contract_price",
                    "roll_state",
                    "second_contract",
                    "second_contract_price",
                    "largest_open_interest_contract",
                    "index",
                ],
            )
            if df2.shape[0] > 30:
                df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
                df3 = pd.concat([df3.iloc[1:, :], new_row], ignore_index=True)
            else:
                if df2.empty:
                    df2 = df1.copy()
                else:
                    df2 = pd.concat([df2, df1], ignore_index=True)
                if df3.empty:
                    df3 = new_row.copy()
                else:
                    df3 = pd.concat([df3, new_row], ignore_index=True)

        elif df3.shape[0] <= 2:
            pre_main_contract = df3["main_contract"].values[-1]
            pre_main_contract_price = df3["main_contract_price"].values[-1]
            pre_second_contract = df3["second_contract"].values[-1]
            pre_second_contract_price = df3["second_contract_price"].values[-1]
            pre_index = df3["index"].values[-1]

            roll_state = 0
            main_contract = pre_main_contract
            main_contract_price = df1[df1["instrument_id"] == main_contract][
                "close"
            ].values[0]
            largest_open_interest_contract = df1["instrument_id"].values[0]
            second_contract = ""
            second_contract_price = 0
            index = pre_index * (main_contract_price / pre_main_contract_price)

            list1 = [
                date1,
                main_contract,
                main_contract_price,
                roll_state,
                second_contract,
                second_contract_price,
                largest_open_interest_contract,
                index,
            ]
            with open(path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(list1)

            new_row = pd.DataFrame(
                [list1],
                columns=[
                    "date",
                    "main_contract",
                    "main_contract_price",
                    "roll_state",
                    "second_contract",
                    "second_contract_price",
                    "largest_open_interest_contract",
                    "index",
                ],
            )
            if df2.shape[0] > 30:
                df2 = pd.concat([df2.iloc[1:, :], df1], ignore_index=True)
                df3 = pd.concat([df3.iloc[1:, :], new_row], ignore_index=True)
            else:
                if df2.empty:
                    df2 = df1.copy()
                else:
                    df2 = pd.concat([df2, df1], ignore_index=True)
                if df3.empty:
                    df3 = new_row.copy()
                else:
                    df3 = pd.concat([df3, new_row], ignore_index=True)

        else:  # 按照南华指数的展期规则进行展期
            main_contract = df3["main_contract"].values[-1]
            roll_state = df3["roll_state"].values[-1]
            largest_open_interest_contract = df3[
                "largest_open_interest_contract"
            ].values[-1]
            roll_condition = ci.cal_date_spread(
                pd.Timestamp(date1),
                ci.get_maturity_date(main_contract, pd.Timestamp(date1)),
                pd.to_datetime(sorted(set(df["calender_day"]))),
            )
            if (
                (roll_state == 0)
                & (roll_condition["months_remaining"] <= 2)
                & (roll_condition["trading_days_left_this_month"] <= 5)
            ):
                # 需要强制展期时
                df2, df3 = roll_calculate_function(df1, df2, df3, date1, path)
            else:
                if (
                    (roll_state == 0)
                    & (len(set(df3.tail(3)["largest_open_interest_contract"])) == 1)
                    & (main_contract != largest_open_interest_contract)
                    & (
                        ci.get_maturity_date(
                            largest_open_interest_contract, pd.Timestamp(date1)
                        )
                        > ci.get_maturity_date(main_contract, pd.Timestamp(date1))
                    )
                ):
                    # 当前展期状态为0, 连续三天最大持仓合约是同一合约且不是主力合约, 主力合约比持仓量最大的合约到期日近
                    df2, df3 = roll_calculate_function(df1, df2, df3, date1, path)
                else:
                    df2, df3 = normal_calculate_function(df1, df2, df3, date1, path)


if __name__ == "__main__":
    code_list1 = ["A", "AG", "AL", "AP", "AU", "C", "CF", "CU", "EG", "FG", "I", "J"]
    code_list2 = ["IH", "IF", "IC", "IM", "TS", "TF", "T", "TL"]
    for code1 in code_list1:
        path1 = (
            "D:\\LearningAndWorking\\VSCode\\python\\project1\\" + code1 + ".csv"
        )  # 数据输出路径
        list1 = [
            "date",
            "main_contract",
            "main_contract_price",
            "roll_state",
            "second_contract",
            "second_contract_price",
            "largest_open_interest_contract",
            "index",
        ]
        with open(path1, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(list1)

        main(code1, path1)
