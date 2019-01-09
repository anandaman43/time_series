from selection import load_data
from data_transformation import get_weekly_aggregate
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil import parser
from outlier import ma_replace_outlier
from tqdm import tqdm
from smoothing import *
import warnings
warnings.filterwarnings("ignore")


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
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = load_data()
    overall = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_cleaveland.csv")
    k = 0
    for index, row in tqdm(overall.iterrows()):
        frequency = row["frequency"]
        days = row["days"]
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        if frequency == 0:
            continue
        df_series = get_weekly_aggregate(df_series)
        _testing = df_series[["quantity", "dt_week"]].copy()
        aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

        aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
        aggregated_data.y = aggregated_data.y.apply(float)
        aggregated_data = aggregated_data.sort_values('ds')
        aggregated_data = aggregated_data.reset_index(drop=True)
        outlier = True
        if (frequency >= 26) & (days > 365 + 183):
            n_pass = 3
            window_size = 12
            sigma = 4.0
        elif(frequency >= 20) & (frequency < 26):
            n_pass = 3
            window_size = 12
            sigma = 5.0
        elif (frequency >= 26) & (days <= 365+183):
            if len(aggregated_data) >= 26:
                n_pass = 3
                window_size = 12
                sigma = 4.0
            else:
                outlier = False
        elif (frequency >= 12) & (frequency < 20):
            if len(aggregated_data) >= 26:
                n_pass = 3
                window_size = 24
                sigma = 5.0
            else:
                outlier = False
        elif frequency < 12:
            outlier = False

        if outlier:
            _result = ma_replace_outlier(data=aggregated_data, n_pass=n_pass, aggressive=True, window_size=window_size,
                                         sigma=sigma)
            result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        else:
            result = aggregated_data.rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        if k == 1:
            final = pd.concat([final, result])
        if k == 0:
            final = result
            k = 1
    final = final.groupby("dt_week")["quantity"].sum().reset_index()
    final = final.set_index("dt_week")
    final.to_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed.csv")
    result = seasonal_decompose(final["quantity"], model="additive")
    result.seasonal.to_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed_seas.csv")
    return result.seasonal


def overall_aggregate_seas_2():
    season = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed_seas.csv",
    names=["dt_week", "quantity"])
    season["dt_week"] = season["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    season = season.set_index("dt_week")
    return season


def overall_aggregate_seas_3_point():
    season = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed_seas.csv",
    names=["dt_week", "quantity"])
    season = smoothing_3(season)
    season["dt_week"] = season["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    season = season.set_index("dt_week")
    return season


if __name__ == "__main__":
    # overall_aggregate_seas()
    print(overall_aggregate_seas_2().head())
    print(overall_aggregate_seas_3_point().head())
