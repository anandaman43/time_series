from selection import load_data
from data_transformation import get_weekly_aggregate
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from stldecompose import decompose, forecast
from dateutil import parser
from outlier import ma_replace_outlier
from smoothing import *
import warnings
warnings.filterwarnings("ignore")


def stores_analysis(input_df, matnr=103029):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_cleaveland.csv")
    overall = overall[overall["matnr"] == matnr]
    product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    print(product[product["matnr"] == str(matnr)]["description"].values[0])
    product_name = product[product["matnr"] == str(matnr)]["description"].values[0]
    k = 0
    for index, row in overall.iterrows():
        frequency = row["frequency"]
        days = row["days"]
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        if int(frequency) == 0:
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
    try:
        #final = final.groupby("dt_week")["quantity"].sum().reset_index()
        return final
    except:
        return None


if __name__=="__main__":
    df = load_data()
    df = df [df["quantity"] >= 0]
    df = df[df["matnr"] == 101728]
    series_stores = pd.DataFrame()
    for index, group in df.groupby(["date"]):
        num = len(group["kunag"])
        series_stores = series_stores.append([[index, num]])
    series_stores.columns = ["date", "quantity"]
    series_stores = series_stores[series_stores["date"] >= 20160703]
    series_stores["matnr"] = "A"
    series_stores["kunag"] = "B"
    series_stores["price"] = 0
    series_stores["date"] = series_stores["date"].map(str)
    series_stores = series_stores.reset_index(drop=True)
    series_stores = get_weekly_aggregate(series_stores)
    series_stores = series_stores.set_index("dt_week")
    series_stores.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/stores_analysis.csv")
    plt.plot(series_stores["quantity"], marker=".")
    plt.show()

    data_stores = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/stores_analysis.csv")
    data_aggregated = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/aggregated.csv")
    data_aggregated["division"] = data_aggregated["quantity"]/data_stores["quantity"]
    data_aggregated = data_aggregated.set_index("dt_week")
    plt.plot(data_aggregated["division"], marker=".")
    plt.show()




