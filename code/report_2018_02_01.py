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
import matplotlib.pyplot as plt
import time


df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
report = pd.DataFrame()
count = 0
for index, row in sample.iterrows():
    start = time.time()
    kunag = int(row["kunag"])
    matnr = int(row["matnr"])
    seas_pres = ljung_box_test(df, int(row["matnr"]))
    if not seas_pres[0]:
        count += 1
        print(count)
        continue
    else:
        dtw_flag = dtw_check(df, kunag, matnr)
        seasonality_product = product_seasonal_comp_7_point(df, matnr)
        df_series = individual_series(df, kunag, matnr)
        result_1 = arima_seasonality_added_rolling(df_series, seasonality_product)
        result_1_011 = arima_seasonality_added_rolling_011(df_series, seasonality_product)
        result_1_022 = arima_seasonality_added_rolling_022(df_series, seasonality_product)
        result_1 = result_1.set_index("dt_week")
        result_2 = arima_rolling(df_series)
        result_2 = result_2.set_index("dt_week")
        # plt.figure(figsize=(16, 8))
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('time (weekly)')
        ax1.set_ylabel('actual quantities', color=color)
        ax1.plot(result_1["quantity"], marker=".", markerfacecolor="red", color=color, label="actual")
        # ax1.plot(result_1_011["prediction"].iloc[-16:], marker=".", markerfacecolor="red", color="black",
        #          label="prediction_seasonal_011")
        # ax1.plot(result_1_022["prediction"].iloc[-16:], marker=".", markerfacecolor="red", color="brown",
        #          label="prediction_seasonal_022")
        ax1.plot(result_1["prediction"].iloc[-16:], marker=".", markerfacecolor="red", color="green", label="prediction_seasonal")
        ax1.plot(result_2["prediction"].iloc[-16:], marker=".", markerfacecolor="red",  color="orange", label="prediction_normal")
        ax1.tick_params(axis='y', labelcolor=color)
        # seas_error = pow(mean_squared_error(result_1.iloc[-16:]["quantity"], result_1.iloc[-16:]["prediction"]), 0.5)
        # norm_error = pow(mean_squared_error(result_2.iloc[-16:]["quantity"], result_2.iloc[-16:]["prediction"]), 0.5)
        # plt.title("seas_error: " + str(seas_error) + "  norm_error: " + str(norm_error))
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.xlabel("Date", fontsize=14)
        # plt.ylabel("Quantity", fontsize=14)
        # plt.legend()
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('seasonal quantities', color=color)  # we already handled the x-label with ax1
        ax2.plot(seasonality_product, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        #plt.legend()
        fig.tight_layout()
        end = time.time()
        print("time consumed :", (end - start)/60, "minutes")
        #plt.legend()
        plt.show()
        #plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_18/"
                 #   + str(kunag) + "_" + str(matnr) + "_" + str(dtw_flag) + ".png")

        seas_error = 0
        norm_error = 0
        report = report.append([[kunag, matnr, seas_pres[1][0], dtw_flag[1], seas_error, norm_error]])
        pass
    count += 1
    print(count)
report.columns = ["kunag", "matnr", "seasonality", "dtw_flag", "seas_error", "norm_error"]
report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_25/report_2018_02_25.csv")
    # except:
    #     pass