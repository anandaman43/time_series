__doc__ = """this code is written to check if similarity threshold can be moved from 0.18 to 0.20 in category 1, 2, 3 
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
import time
import os


project = "/home/aman/PycharmProjects/seasonality_hypothesis/"
folder = "category_2_ver1"
file = "report.csv"
folder_address = os.path.join(project, folder)
file_address = os.path.join(folder_address, file)
os.mkdir(folder_address)

df = load_data()
frequency_cleaveland = pd.read_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
# bucket_1_sample = frequency_cleaveland[(frequency_cleaveland["frequency"] > 26) & (frequency_cleaveland["days"] > 730)].sample(400, random_state=1)
# bucket_1_sample.to_csv(
#     "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv", index=False)
sample = frequency_cleaveland[(frequency_cleaveland["frequency"] >= 12) & (frequency_cleaveland["frequency"] < 20)
                              & (frequency_cleaveland["days"] > 112)]
print("Number of combos: ", sample.shape[0])
sample = sample.sample(400, random_state=1)
sample.to_csv(folder_address+"/sample.csv")
sample = pd.read_csv(folder_address+"/sample.csv")
report = pd.DataFrame()
count = 0
# start = time.time()
for index, row in sample.iterrows():
    kunag = int(row["kunag"])
    matnr = int(row["matnr"])
    try:
        test1 = ljung_box_test(df, matnr)
        test1_flag = test1[0]
        test1_pvalue = test1[1]
        aggregated_product = test1[4]
        test2 = dtw_check(df, kunag, matnr)
        test2_flag = test2[0]
        test2_value = test2[1]
    except:
        test1 = False
        test2 = False
    df_series = individual_series(df, kunag, matnr)
    if test1_flag:
        seasonality_component = product_seasonal_comp_7_point(df, matnr)
        result_52_seasonal = moving_average_with_seasonality(df_series, seasonality_component, order=52)
        result_52_nonseasonal = moving_average(df_series, order=52)
        # result = arima_rolling_011(df_series)
        # result_seasonal = arima_seasonality_added_rolling_011(df_series, seasonality_component)
        error_result_nonseasonal = pow(mean_squared_error(df_series["quantity"].iloc[-16:],
                                              result_52_nonseasonal["prediction"].iloc[-16:]), 0.5)
        error_result_seasonal = pow(mean_squared_error(df_series["quantity"].iloc[-16:],
                                                       result_52_seasonal["prediction"].iloc[-16:]), 0.5)
        report = report.append([[kunag, matnr, test1_flag, test1_pvalue, test2_flag, test2_value,
                                 error_result_nonseasonal, error_result_seasonal]])
        report.to_csv(file_address, index=False)
        count += 1
        print("count: ", count)
        df_series = df_series.set_index("dt_week")
        plt.figure(figsize=(16, 8))
        plt.subplot(311)
        plt.plot(df_series, marker=".", markerfacecolor="red", label="actual")
        plt.plot(result_52_nonseasonal.set_index("dt_week").iloc[-16:]["prediction"], marker=".", markerfacecolor="red",
                 label="order_52_" + str(error_result_nonseasonal))
        plt.plot(result_52_seasonal.iloc[-16:]["prediction"], marker=".", markerfacecolor="red",
                 label="order_52_seasonal_" + str(error_result_seasonal))
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        plt.subplot(312)
        plt.plot(aggregated_product.set_index("dt_week")["quantity"], marker=".", markerfacecolor="red", label="aggregated")
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        plt.subplot(313)
        print(seasonality_component)
        plt.plot(seasonality_component["quantity"], marker=".", markerfacecolor="red", label="seasonality")
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        # plt.show()
        plt.title(str(test2_flag))
        plt.savefig(folder_address + "/" + str(test2_flag) + "_" + str(kunag) + "_" + str(matnr)+".png")
report.columns = ["kunag", "matnr", "test1_flag", "test1_pvalue", "test2_flag", "test2_value", "error_result",
                  "error_result_seasonal"]

# end = time.time()
# print("time consumed :", (end - start) / 60, " minutes")
report.to_csv(file_address, index=False)

"""              
report = pd.read_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/category_8/report.csv")
report["min_order"] = report.apply(lambda x: x.argmin(x), axis=1)
print(report["min_order"].value_counts())
"""