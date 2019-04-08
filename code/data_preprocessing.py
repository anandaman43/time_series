__doc__ = """creating whole pipeline from raw data to final customer material data"""

import pandas as pd
from selection import load_data
import time
from datetime import datetime
from outlier import ma_replace_outlier
from dateutil import parser
import pickle


def normalized_frequency_and_days(input_df):
    df_copy = input_df.copy()
    df_copy["parse_date"] = pd.to_datetime(df_copy["date"], format="%Y%m%d")
    df_copy = df_copy.sort_values("parse_date")
    if df_copy.shape[0] == 0:
        return 0, 0
    end_date = df_copy["parse_date"].iloc[-1]
    start_date = df_copy["parse_date"].iloc[0]
    num_days = (end_date - start_date).days
    last_year_date = datetime(end_date.year-1, end_date.month, end_date.day)
    if num_days == 0:
        return 0, 0
    if num_days > 365:
        freq = df_copy[(df_copy["parse_date"] >= last_year_date) & (df_copy["parse_date"] <= end_date)].shape[0]
        return freq, num_days
    else:
        freq = df_copy[(df_copy["parse_date"] >= last_year_date) & (df_copy["parse_date"] <= end_date)].shape[0]
        norm_freq = (365/num_days)*freq
        return norm_freq, num_days


def category(frequency, days):
    if frequency >= 26:
        if days > 731 + 183:
            return 1
        if days > 731:
            return 2
        if days > 365 + 183:
            return 3
        else:
            return 7
    if frequency >= 20:
        if days > 731 + 183:
            return 4
        if days > 365 + 183:
            return 5
        else:
            return 8
    if frequency >= 12:
        return 9
    if frequency >= 4:
        return 10


# df = load_data()
# df = df[df["quantity"] >= 0]
# # print(df.head())
# new_df = pd.DataFrame()
# count = 0
# start = time.time()
# current_date = pd.to_datetime("20190105", format="%Y%m%d")
# for index, group in df.groupby(["kunag", "matnr"]):
#     time_series = group[["date", "quantity"]].sort_values("date")
#     time_series_last_date = pd.to_datetime(time_series.iloc[-1]["date"], format="%Y%m%d")
#     diff = (current_date - time_series_last_date).days
#     if diff <= 92:
#         new_df = new_df.append([[index, time_series]])
#     count += 1
#     if count%1000 == 0:
#         print("count: ", count)
# new_df.columns = ["kunag_matnr", "time_series"]
# end = time.time()
# print("time :", end - start)
# print("count :", count)
#
# new_df["freq_and_days"] = new_df["time_series"].apply(lambda x: normalized_frequency_and_days(x))
# new_df["category"] = new_df["freq_and_days"].apply(lambda x: category(x[0], x[1]))
# print(new_df)
#

# pickle_out = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_c005d.pickle", "wb")
# pickle.dump(new_df, pickle_out)
# pickle_out.close()

# pickle_in = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_c005d.pickle", "rb")
# raw_data = pickle.load(pickle_in)
# from data_transformation import get_weekly_aggregate
# def get_weekly_aggregate_(x):
#     x["kunag"] = "A"
#     x["matnr"] = "B"
#     x["price"] = "B"
#     x = get_weekly_aggregate(x)
#     return x
# raw_data["weekly_agg_time_series"] = raw_data["time_series"].apply(lambda x: get_weekly_aggregate_(x))
# pickle_out = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_agg_c005d.pickle", "wb")
# pickle.dump(raw_data, pickle_out)
# pickle_out.close()
# pickle_in = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_agg_c005d.pickle", "rb")
# raw_data = pickle.load(pickle_in)


def outlier(df_series, category):
    flag = False
    if category == 1 or category == 2 or category == 3:
        window_size, sigma, flag = 12, 3, True
    if category == 4 or category == 5 or category == 8:
        window_size, sigma, flag = 12, 5, True
    if category == 7 and df_series.shape[0] >= 26:
        window_size, sigma, flag = 12, 4, True
    if category == 9:
        window_size, sigma, flag = 24, 5, True

    _testing = df_series[["quantity", "dt_week"]].copy()
    aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

    aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
    aggregated_data.y = aggregated_data.y.apply(float)
    aggregated_data = aggregated_data.sort_values('ds')
    aggregated_data = aggregated_data.reset_index(drop=True)
    if not flag:
        aggregated_data = aggregated_data.rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        return aggregated_data
    _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=window_size, sigma=sigma)
    result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
    return result


# raw_data["outlier_removed"] = raw_data.apply(lambda row: outlier(row["weekly_agg_time_series"], row["category"]), axis=1)
# pickle_out = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_agg_out_c005d.pickle", "wb")
# pickle.dump(raw_data, pickle_out)
# pickle_out.close()


def outlier_material(df_series):
    _testing = df_series[["quantity", "dt_week"]].copy()
    aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

    aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
    aggregated_data.y = aggregated_data.y.apply(float)
    aggregated_data = aggregated_data.sort_values('ds')
    aggregated_data = aggregated_data.reset_index(drop=True)

    _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3)
    result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
    return result

# pickle_in = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_agg_out_c005d.pickle", "rb")
# raw_data = pickle.load(pickle_in)
# # print(raw_data["category"].value_counts())
# # print(raw_data.shape[0])
# raw_data = raw_data.dropna(subset=["category"], axis=0)
# raw_data["kunag"], raw_data["matnr"] = raw_data["kunag_matnr"].apply(lambda x: x[0]), raw_data["kunag_matnr"].apply(lambda x: x[1])
# raw_data["length"] = raw_data["outlier_removed"].apply(lambda x: len(x))
# raw_data = raw_data[raw_data["length"] > 4]
# raw_data = raw_data.dropna(subset=["category"], axis=0)
# material_data_df = pd.DataFrame()
# for index, group in raw_data.groupby("matnr"):
#     material_data = pd.concat(group["outlier_removed"].tolist(), axis=0).reset_index(drop=True)
#     material_data = material_data[["quantity", "dt_week"]].groupby("dt_week")["quantity"].sum().reset_index()
#     # print(material_data)
#     material_data = outlier_material(material_data)
#     # print(material_data)
#     material_data_df = material_data_df.append([[index, material_data]])
#
# material_data_df.columns = ["matnr", "material_data_agg"]
# pickle_out = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/material_data_agg_out_c005d.pickle", "wb")
# pickle.dump(material_data_df, pickle_out)
# pickle_out.close()

pickle_in = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/material_data_agg_out_c005d.pickle", "rb")
raw_data = pickle.load(pickle_in)
print(raw_data.head().iloc[0]["matnr"])
print(raw_data.head().iloc[0]["material_data_agg"])

