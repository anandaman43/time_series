import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_rolling_011
from hypothesis_2 import arima_seasonality_added_rolling_011
from hypothesis_2 import arima_rolling
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import os
from local_error import local_rmse


def moving_average(input_df, order=12):
    input_df_copy = input_df.copy()
    input_df_copy["prediction"] = input_df_copy["quantity"]
    for i in range(16, 0, -1):
        input_df_copy["prediction"].iloc[-i] = input_df_copy.iloc[-i-order:-i]["quantity"].mean()
    return input_df_copy


def moving_average_with_seasonality(input_df, product_seasonal_comp_7_point, order=36):
    input_df_copy = input_df.copy().set_index("dt_week")
    seasonality_component_copy = product_seasonal_comp_7_point.copy()
    seasonality_component_copy.columns = ["seasonal_quantity"]
    input_df_copy = pd.concat([input_df_copy, seasonality_component_copy], axis=1).dropna(subset=["quantity"])
    min_seasonal_value = input_df_copy["seasonal_quantity"].min()
    # print(min_seasonal_value)
    input_df_copy["seasonal_quantity"] = input_df_copy["seasonal_quantity"] + abs(min_seasonal_value*2)
    input_df_copy["prediction"] = input_df_copy["quantity"]
    for i in range(16, 0, -1):
        seasonal_rolling_mean = input_df_copy.iloc[-i-order:-i]["seasonal_quantity"].mean()
        # print(input_df_copy)
        f = (input_df_copy.iloc[-i]["seasonal_quantity"] - seasonal_rolling_mean)/seasonal_rolling_mean
        x = input_df_copy.iloc[-i-order:-i]["quantity"].mean()
        input_df_copy["prediction"].iloc[-i] = x + x*f
    return input_df_copy


def moving_average_available(input_df, product_product_seasonal_comp_7_point, order=52):
    """
    prediction of last 16 points using moving average of order 52 .if datapoints available is less than 52, will consid-
    er only those number of points.Decreasing factor is calculated using same number of points from seasonal component.
    :param input_df: customer-product level data
    :param product_product_seasonal_comp_7_point: product level seasonal smoothened data
    :param order: moving average max order
    :return: input_df with one more column with predictions
    """
    input_df_copy = input_df.copy()
    seasonal_copy = product_seasonal_comp_7_point.copy()
    num_weeks = input_df_copy.shape[0]
    # if num_weeks >= 68:
    #
    # else:
    #     order = num_weeks - 16
    return output_df


project = "/home/aman/PycharmProjects/seasonality_hypothesis/"
folder = "local_error"
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
sample = frequency_cleaveland[(frequency_cleaveland["frequency"] >= 20) & (frequency_cleaveland["frequency"] < 26)
                              & (frequency_cleaveland["days"] <= 731 + 183) & (frequency_cleaveland["days"] > 365 + 183)]
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
    if not test2_flag:
        # seasonality_component = product_seasonal_comp_7_point(df, matnr)
        # result_52_seasonal = moving_average_with_seasonality(df_series, seasonality_component, order=52)
        result_09 = moving_average(df_series, order=9)
        result_011 = arima_rolling_011(df_series)
        # result_seasonal = arima_seasonality_added_rolling_011(df_series, seasonality_component)
        error_result_09 = pow(mean_squared_error(df_series["quantity"].iloc[-16:],
                                              result_09["prediction"].iloc[-16:]), 0.5)
        error_result_011 = pow(mean_squared_error(df_series["quantity"].iloc[-16:],
                                                       result_011["prediction"].iloc[-16:]), 0.5)
        local_error_09 = local_rmse(result_09["prediction"].iloc[-16:].values, df_series["quantity"].iloc[-16:].values)
        local_error_011 = local_rmse(result_011["prediction"].iloc[-16:].values, df_series["quantity"].iloc[-16:].values)
        # report = report.append([[kunag, matnr, test1_flag, test1_pvalue, test2_flag, test2_value,
        #                          error_result_09, error_result_011]])
        # report.to_csv(file_address, index=False)
        # count += 1
        # print("count: ", count)
        df_series = df_series.set_index("dt_week")
        plt.figure(figsize=(16, 8))
        # plt.subplot(311)
        plt.plot(df_series, marker=".", markerfacecolor="red", label="actual")
        plt.plot(result_09.set_index("dt_week").iloc[-16:]["prediction"], marker=".", markerfacecolor="red",
                 label="order_12_" + str(error_result_09) + "_" + str(local_error_09))
        plt.plot(result_011.set_index("dt_week").iloc[-16:]["prediction"], marker=".", markerfacecolor="red",
                 label="arima_011_" + str(error_result_011) + "_" + str(local_error_011))
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        plt.show()
        # plt.subplot(312)
        # plt.plot(aggregated_product.set_index("dt_week")["quantity"], marker=".", markerfacecolor="red", label="aggregated")
        # plt.xlabel("Date", fontsize=14)
        # plt.ylabel("Quantity", fontsize=14)
        # plt.legend()
        # # plt.show()
        # plt.title(str(test2_flag))
        # plt.savefig(folder_address + "/" + str(test2_flag) + "_" + str(kunag) + "_" + str(matnr)+".png")
# report.columns = ["kunag", "matnr", "test1_flag", "test1_pvalue", "test2_flag", "test2_value", "error_result_09",
#                   "error_result_011"]

# end = time.time()
# print("time consumed :", (end - start) / 60, " minutes")
# report.to_csv(file_address, index=False)

"""              
report = pd.read_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/category_8/report.csv")
report["min_order"] = report.apply(lambda x: x.argmin(x), axis=1)
print(report["min_order"].value_counts())
"""