from selection import load_data
from data_transformation import get_weekly_aggregate
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil import parser
from outlier import ma_replace_outlier


def aggregate_seasonal_comp():
    df = load_data()[["date", "quantity"]]
    df = df[(df["quantity"] >= 0) & (df["quantity"] <= 10)]
    aggregate_data = df.groupby("date")["quantity"].sum()
    aggregate_data = aggregate_data.reset_index()
    aggregate_data["kunag"] = 1
    aggregate_data["matnr"] = 2
    aggregate_data["price"] = 3
    aggregate_data = get_weekly_aggregate(aggregate_data)
    aggregate_data["dt_week"] = aggregate_data["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    aggregate_data = aggregate_data.set_index("dt_week")
    # plt.figure(figsize=(16, 8))
    # plt.plot(aggregate_data["quantity"], label='quantity')
    # plt.title("aggregated plot")
    # plt.show()
    result = seasonal_decompose(aggregate_data["quantity"], model="additive")
    # result.plot()
    plt.show()
    return result.seasonal


def aggregate_seasonal_comp_2():
    df_2 = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/groupby_aggregated_outlier_removed.csv",
        names=["dt_week", "quantity"])
    df_2["dt_week"] = df_2["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    aggregate_data = df_2.set_index("dt_week")
    result = seasonal_decompose(aggregate_data["quantity"], model="additive")
    return result.seasonal


def samples_aggregate_seas():
    df = load_data()
    bucket_1_sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
    k = 0
    for index, row in bucket_1_sample.iterrows():
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        df_series = get_weekly_aggregate(df_series)
        _testing = df_series[["quantity", "dt_week"]].copy()
        aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

        aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
        aggregated_data.y = aggregated_data.y.apply(float)
        aggregated_data = aggregated_data.sort_values('ds')
        aggregated_data = aggregated_data.reset_index(drop=True)

        _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
        result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        if k == 1:
            final = pd.concat([final, result])
        if k == 0:
            final = result
            k = 1
    final = final.groupby("dt_week")["quantity"].sum().reset_index()
    final = final.set_index("dt_week")
    #plt.figure(figsize=(16, 8))
    #plt.plot(final["quantity"], label='quantity', marker=".")
    #plt.title("200 sample aggregated plot")
    #plt.xlabel("dt_weeks")
    #plt.ylabel("aggregated quantities")
    #plt.show()
    result = seasonal_decompose(final["quantity"], model="additive")
    #result.plot()
    #plt.show()
    return result.seasonal


def overall_aggregate_seas():
    df = load_data()
    bucket_1_sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
    k = 0
    for index, row in bucket_1_sample.iterrows():
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        df_series = get_weekly_aggregate(df_series)
        _testing = df_series[["quantity", "dt_week"]].copy()
        aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

        aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
        aggregated_data.y = aggregated_data.y.apply(float)
        aggregated_data = aggregated_data.sort_values('ds')
        aggregated_data = aggregated_data.reset_index(drop=True)
        if case1 2 3:
            n_pass = 3
            window_size = 12
            sigma = 4.0
        elif(case4 5 6 8):
            n_pass = 3
            window_size = 12
            sigma = 5.0
        elif case 7:
            if



        _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
        result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        if k == 1:
            final = pd.concat([final, result])
        if k == 0:
            final = result
            k = 1
    final = final.groupby("dt_week")["quantity"].sum().reset_index()
    final = final.set_index("dt_week")
    #plt.figure(figsize=(16, 8))
    #plt.plot(final["quantity"], label='quantity', marker=".")
    #plt.title("200 sample aggregated plot")
    #plt.xlabel("dt_weeks")
    #plt.ylabel("aggregated quantities")
    #plt.show()
    result = seasonal_decompose(final["quantity"], model="additive")
    #result.plot()
    #plt.show()
    return result.seasonal


if __name__=="__main__":

    # df_2 = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/groupby_aggregated_outlier_removed.csv",
    #                    names=["dt_week", "quantity"])
    # df_2["dt_week"] = df_2["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    # aggregate_data = df_2.set_index("dt_week")
    # plt.figure(figsize=(16, 8))
    # plt.plot(aggregate_data["quantity"], label='quantity')
    # plt.title("aggregated plot")
    # plt.show()
    # result = seasonal_decompose(aggregate_data["quantity"], model="additive")
    # result.plot()
    # plt.show()
    # print(aggregate_seasonal_comp())
    plt.plot(samples_aggregate_seas(), marker=".")
    plt.xlabel("dt_weeks")
    plt.ylabel("seasonal component (aggregated)")
    plt.title("seasonal plot")
    plt.show()
    aggregate_seasonal_comp()