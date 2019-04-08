import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_seasonality_added_rolling_011
from hypothesis_2 import arima_seasonality_added_rolling_022
from hypothesis_2 import arima_rolling
from hypothesis_2 import arima_rolling_011
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time


df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_12_20_400_sample.csv")
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
        seasonality_product = product_seasonal_comp_7_point(df, matnr).iloc[-55:-3]
        df_series = individual_series(df, kunag, matnr)
        result_1 = arima_seasonality_added_rolling_011(df_series, seasonality_product)
        break
        result_1 = result_1.set_index("dt_week")
        result_2 = arima_rolling_011(df_series)
        result_2 = result_2.set_index("dt_week")
        plt.figure(figsize=(16, 8))
        plt.plot(result_1["prediction"].iloc[-16:], marker=".", markerfacecolor="red", label="prediction_seasonal")
        plt.plot(result_2["prediction"].iloc[-16:], marker=".", markerfacecolor="red", label="prediction_normal")
        plt.plot(result_1["quantity"], marker=".", markerfacecolor="red", label="actual")
        seas_error = pow(mean_squared_error(result_1.iloc[-16:]["quantity"], result_1.iloc[-16:]["prediction"]), 0.5)
        norm_error = pow(mean_squared_error(result_2.iloc[-16:]["quantity"], result_2.iloc[-16:]["prediction"]), 0.5)
        plt.title("seas_error: " + str(seas_error) + "  norm_error: " + str(norm_error))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        end = time.time()
        print("time consumed :", (end - start)/60, "minutes")
        # plt.show()
        plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/report_2019_03_04/"
                   + str(kunag) + "_" + str(matnr) + "_" + str(dtw_flag) + ".png")
        report = report.append([[kunag, matnr, seas_pres[1][0], dtw_flag[1], seas_error, norm_error]])
        report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2019_03_04/report_2019_03_04.csv")
    count += 1
    print(count)
# report.columns = ["kunag", "matnr", "seasonality", "dtw_flag", "seas_error", "norm_error"]
# report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2019_03_04/report_2019_03_04.csv")
    # except:
    #     pass