# from outlier import ma_replace_outlier
# from selection import load_data
# from selection import remove_negative_rows
# from data_transformation import get_weekly_aggregate
# from outlier import ma_replace_outlier
# from dateutil import parser
# import pandas as pd
# df = load_data()
# df = remove_negative_rows(df)
# df = df[df["date"] >= 20160704]
# df = get_weekly_aggregate(df)
# _testing = df[["quantity", "dt_week"]].copy()
# aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})
# aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
# aggregated_data.y = aggregated_data.y.apply(float)
# aggregated_data = aggregated_data.sort_values('ds')
# aggregated_data = aggregated_data.reset_index(drop=True)
# _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
# _result[0].to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_outlier_removed.csv")
# _result[0].groupby("ds")["y"].sum().to_csv(
#     "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/groupby_aggregated_outlier_removed.csv")

"""
import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_rolling
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
report = pd.DataFrame()
count = 0
for index, row in sample.iterrows():
    kunag = 500078638
    matnr = 145277
    try:
        seas_pres = ljung_box_test(df, int(row["matnr"]))[0]
        if not seas_pres:
            count += 1
            continue
        else:
            dtw_flag = dtw_check(df, kunag, matnr)
            seasonality_product = product_seasonal_comp_7_point(df, matnr)
            df_series = individual_series(df, kunag, matnr)
            result_1 = arima_seasonality_added_rolling(df_series, seasonality_product)
            result_1 = result_1.set_index("dt_week")
            result_2 = arima_rolling(df_series)
            result_2 = result_2.set_index("dt_week")
            plt.figure(figsize=(16, 8))
            plt.plot(result_1["prediction"], marker=".", label="prediction_seasonal")
            plt.plot(result_2["prediction"], marker=".", label="prediction_normal")
            plt.plot(result_1["quantity"], marker=".", label="actual")
            seas_error = pow(mean_squared_error(result_1.iloc[-16:]["quantity"], result_1.iloc[-16:]["prediction"]), 0.5)
            norm_error = pow(mean_squared_error(result_2.iloc[-16:]["quantity"], result_2.iloc[-16:]["prediction"]), 0.5)
            plt.title("seas_error: " + str(seas_error) + "  norm_error: " + str(norm_error))
            plt.legend()
            plt.show()
            report = report.append([[kunag, matnr, seas_pres, dtw_flag, seas_error, norm_error]])
    except:
        count += 1
        pass
    count += 1
    print(count)

"""

# import pandas as pd
# d = {"A": [1,1,3], "B" : [4,5,6]}
# df = pd.DataFrame(d)
# print(df)
# def printa(x):
#     print(x)
#     return True
#
# print(df.filter(lambda x:printa(x)))

import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_seasonality_added_rolling_011
from hypothesis_2 import arima_seasonality_added_rolling_022
from hypothesis_2 import arima_rolling
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time

# def moving_average(input_df, order=12):
#     input_df_copy = input_df.copy()
#     input_df_copy["prediction"] = input_df_copy["quantity"]
#     for i in range(16, 0, -1):
#         input_df_copy["prediction"].iloc[-i] = input_df_copy.iloc[-i-order:-i]["quantity"].mean()
#     return input_df_copy
#
#
# df = load_data()
# sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_0_12_sample.csv")
# report = pd.DataFrame()
# count = 0
# for index, row in sample.iterrows():
#     count += 1
#     if count < 191:
#         continue
#     start = time.time()
#     kunag = int(row["kunag"])
#     matnr = int(row["matnr"])
#
#     df_series = individual_series(df, kunag, matnr)
#     result_12 = moving_average(df_series, order=12)
#     error_12 = pow(mean_squared_error(df_series["quantity"].iloc[-16:], result_12["prediction"].iloc[-16:]), 0.5)
#
#     end = time.time()
#
#     print("count: ", count, "time:", (end-start)/60)
#     result_12 = result_12.set_index("dt_week")
#     df_series = df_series.set_index("dt_week")
#     plt.figure(figsize=(16, 8))
#     plt.plot(df_series, marker=".", markerfacecolor="red", label="actual")
#     plt.plot(result_12.iloc[-16:]["prediction"], marker=".", markerfacecolor="red", label="prediction")
#     plt.title("rmse_error: " + str(error_12))
#     plt.xlabel("Date", fontsize=14)
#     plt.ylabel("Quantity", fontsize=14)
#     plt.legend()
#     plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/ma_report_2018_02_27_012/"+ str(kunag)+"_"+str(matnr)+".png")
    # plt.plot(result_2["prediction"].iloc[-16:], marker=".", markerfacecolor="red", label="prediction_normal")
    # plt.plot(result_1["quantity"], marker=".", markerfacecolor="red", label="actual")
    # result_1, orders = arima_seasonality_added_rolling(df_series, seasonality_product)
# report.columns = ["kunag", "matnr", error_04, error_05, error_06, error_07, error_08, error_09, error_10,
#                              error_11, error_12, error_13, error_14, error_15, error_16, error_17, error_18, error_19,
#                              error_20, error_24]


# report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/ma_report_2018_02_27/report_2018_02_27.csv",
#               index=False)
#
# report = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/ma_report_2018_02_27/report_2018_02_27.csv")
# report["min_order"] = report.apply(lambda x: x.argmin(x), axis=1)
# print(report["min_order"].value_counts())
"""

report = pd.read_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/52_available/2019_03_08.csv")
report = report[["error_52", "error_06"]]

report["min_order"] = report.apply(lambda x: x.argmin(x), axis=1)
print(report["min_order"].value_counts())



import pandas as pd
from dtw import dtw
d1 = {"quantity": [1, 2, 3, 4, 5], "date" : [1, 2, 3, 4, 5]}
d2 = {"quantity2": [-1, -2, -3, 2, 7], "date" : [2, 3, 4, 9, 10]}
df1 = pd.DataFrame(d1)
df1 = pd.concat([df1, df1], axis=0).reset_index()
df2 = pd.DataFrame(d2)
df2 = pd.concat([df2, df2], axis=0).reset_index()
df2["date"].iloc[-1] = 5
print(df1)
print(df2)
l2_norm = lambda x, y: (x - y) ** 2
x = df1["date"]
y = df2["date"]
print(dtw(x, y, dist=l2_norm)[0])
# a = pd.concat([df1, df2], axis=1).reset_index()
# b = a.dropna(subset=["quantity2"])
# start, end = b.index[0], b.index[-1]
# print(start, end)


"""
# import pandas as pd
# frequency_cleaveland = pd.read_csv(
#     "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
# # bucket_1_sample = frequency_cleaveland[(frequency_cleaveland["frequency"] > 26) & (frequency_cleaveland["days"] > 730)].sample(400, random_state=1)
# # bucket_1_sample.to_csv(
# #     "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv", index=False)
# sample = frequency_cleaveland[(frequency_cleaveland["frequency"] >= 26) & (frequency_cleaveland["days"] > 92) &
#                               (frequency_cleaveland["days"] <= 365 + 183)]
# sample = sample["matnr"]
# print(sample.shape)
# import profile
from selection import load_data
# profile.run("load_data()")
# df = load_data()
# print(df.shape)
# print(df.head())
# df = df.groupby(["kunag", "matnr"])["quantity"].sum()
# print(df.shape)

# from selection import load_data
# df = load_data()
# print(df.sort_values("date", ascending=True))
#
# report = pd.read_csv(
#     "/home/aman/PycharmProjects/seasonality_hypothesis/category_8/report.csv")
# report = report[["order_06", "order_09"]]
# report["min_order"] = report.apply(lambda x: x.argmin(x), axis=1)
# print(report["min_order"].value_counts())

import numpy as np
import pandas as pd

# def moving_average(data, window_size):
#     """ Computes moving average using discrete linear convolution of two one dimensional sequences.
#     Args:
#     -----
#             data (pandas.Series): independent variable
#             window_size (int): rolling window size
#
#     Returns:
#     --------
#             ndarray of linear convolution
#
#     References:
#     ------------
#     [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
#     [2] API Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
#
#     """
#     import numpy as np
#     window = np.ones(int(window_size)) / float(window_size)
#     if (window_size % 2 == 0):  # even
#         beg = int(window_size/2)
#         end = int(window_size/2) - 1
#         mean_beg = np.array([np.mean(data[:window_size])] * beg)
#         mean_end = np.array([np.mean(data[:window_size])] * end)
#     else: #odd
#         beg = int((window_size - 1) / 2)
#         end = int((window_size - 1) / 2)
#         mean_beg = np.array([np.mean(data[:window_size])] * beg)
#         mean_end = np.array([np.mean(data[:window_size])] * end)
#     data = np.concatenate([mean_beg, np.array(data), mean_end])
#
#     return np.convolve(data, window, 'valid')
#
#
#
data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/115584.csv")


y = data.quantity
from outlier import ma_replace_outlier
from dateutil import parser
aggregated_df = data
print(aggregated_df)
_testing = aggregated_df[["quantity", "dt_week"]].copy()
aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
aggregated_data.y = aggregated_data.y.apply(float)
aggregated_data = aggregated_data.sort_values('ds')
aggregated_data = aggregated_data.reset_index(drop=True)
n_pass = 3
window_size = 12
sigma = 3.0
_result = ma_replace_outlier(data=aggregated_data, n_pass=n_pass, aggressive=True, window_size=window_size,
                             sigma=sigma)
result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
print(result.iloc[31:34])

# import numpy as np
# a = np.ones(10)
#
# a[7] = 2
# print(a)
# print(np.where(a == 2))