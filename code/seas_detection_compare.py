import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from seasonality_detection import ljung_box_test_without_aggregation
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_rolling
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = load_data()
result = pd.DataFrame()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
count = 1
for matnr in sample["matnr"].unique():
    seas_pres_1 = ljung_box_test(df, matnr)
    # seas_pres_2 = ljung_box_test_without_aggregation(df, matnr)
    plt.figure(figsize=(16, 8))
    plt.plot(seas_pres_1[3], marker=".", label="aggregated")
    plt.legend()
    plt.title("ljung p_value: " + str(seas_pres_1[1]) + "  dickey p_value: " + str(seas_pres_1[2]))
    plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/seas_detection_compare_2018_02_08_ver_1/"+str(matnr)+".png")
    result = result.append([[matnr, seas_pres_1[1][0], seas_pres_1[2]]])
    print(count)
    count += 1
result.columns = ["matnr", "ljung_p__value", "dickey_p__value"]
result.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/seas_detection_compare_2018_02_08_ver_1/result.csv", index=False)
#         if not seas_pres:
#             continue
#         else:
#             dtw_flag = dtw_check(df, kunag, matnr)
#
#             df_series = individual_series(df, kunag, matnr)
#             result_1 = arima_seasonality_added_rolling(df_series, seasonality_product)
#             result_1 = result_1.set_index("dt_week")
#             result_2 = arima_rolling(df_series)
#             result_2 = result_2.set_index("dt_week")
#
#             plt.plot(result_2["prediction"], marker=".", label="prediction_normal")
#             plt.plot(result_1["quantity"], marker=".", label="actual")
#             seas_error = pow(mean_squared_error(result_1.iloc[-16:]["quantity"], result_1.iloc[-16:]["prediction"]), 0.5)
#             norm_error = pow(mean_squared_error(result_2.iloc[-16:]["quantity"], result_2.iloc[-16:]["prediction"]), 0.5)
#             plt.title("seas_error: " + str(seas_error) + "  norm_error: " + str(norm_error))
#             plt.legend()
#             plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_04/"
#                         + str(kunag) + "_" + str(matnr) + "_" + str(dtw_flag) + ".png")
#             report = report.append([[kunag, matnr, seas_pres, dtw_flag, seas_error, norm_error]])
#     except:
#         pass
#     count += 1
#     print(count)
# report.columns = ["kunag", "matnr", "seasonality", "dtw_flag", "seas_error", "norm_error"]
# report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_04/report_2018_02_04.csv")