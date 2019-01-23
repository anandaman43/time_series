from selection import load_data
from data_transformation import get_weekly_aggregate
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil import parser
from outlier import ma_replace_outlier
from smoothing import *
import warnings
warnings.filterwarnings("ignore")


def smoothing_5(df):
    df_copy = df.copy()
    max_index = df.shape[0] - 1
    for i in range(0, max_index-3):
        mean = df.iloc[i:i+5]["quantity"].mean()
        df_copy["quantity"].iloc[i+2] = mean
    df_copy["quantity"].iloc[0] = df.iloc[0:2]["quantity"].mean()
    df_copy["quantity"].iloc[1] = df.iloc[0:3]["quantity"].mean()
    df_copy["quantity"].iloc[-1] = df.iloc[-2:]["quantity"].mean()
    df_copy["quantity"].iloc[-2] = df.iloc[-3:]["quantity"].mean()
    return df_copy


def product_seasonal_comp(input_df, matnr=103029):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
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
        final = final.groupby("dt_week")["quantity"].sum().reset_index()
    except:
        return None
    final = final.set_index("dt_week")
    result = seasonal_decompose(final["quantity"], model="additive")
    result.plot()
    plt.show()
    product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    product_name = product[product["matnr"] == str(matnr)]["description"].values[0]
    # plt.figure(figsize=(16, 8))
    # plt.plot(result.seasonal, marker=".")
    # plt.title("original_" + product_name)
    # plt.show()
    return result.seasonal


def product_seasonal_comp_5_point(df, matnr):
    season = product_seasonal_comp(df, matnr).reset_index()
    season.columns = ["dt_week", "quantity"]
    season = smoothing_5(season)
    season["dt_week"] = season["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    season = season.set_index("dt_week")
    return season


def product_seasonal_comp_7_point(df, matnr):
    input_df = product_seasonal_comp(df, matnr)
    # plt.plot(input_df, marker=".")
    # plt.title("original")
    # plt.show()
    input_df = input_df.reset_index().copy()
    max_index = input_df.shape[0] - 1
    df_copy = input_df.copy()
    for i in range(0, max_index-5):
        mean = input_df.iloc[i:i+7]["quantity"].mean()
        df_copy["quantity"].iloc[i+3] = mean
    df_copy["quantity"].iloc[0] = df_copy.iloc[0+52]["quantity"]
    df_copy["quantity"].iloc[1] = df_copy.iloc[1+52]["quantity"]
    df_copy["quantity"].iloc[2] = df_copy.iloc[2+52]["quantity"]
    df_copy["quantity"].iloc[-1] = df_copy.iloc[-1-52]["quantity"]
    df_copy["quantity"].iloc[-2] = df_copy.iloc[-2-52]["quantity"]
    df_copy["quantity"].iloc[-3] = df_copy.iloc[-3-52]["quantity"]
    output_df = df_copy.set_index("dt_week")
    return output_df


if __name__=="__main__":
    df = load_data()
    matnr = 112260
    temp = product_seasonal_comp_7_point(df, matnr).reset_index()
    plt.plot(temp.set_index("dt_week"), marker=".")
    plt.title("smoothened")
    plt.show()
    # for i in range(0, 30):
    #     print(temp.iloc[i]["quantity"], temp.iloc[i+52]["quantity"])
    # plt.plot(product_seasonal_comp(df, matnr), marker=".")
    # product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    # product_name = product[product["matnr"] == str(matnr)]["description"].values[0]
    # plt.title("smoothing_" + product_name)
    # plt.show()

