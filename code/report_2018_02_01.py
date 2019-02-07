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
    kunag = int(row["kunag"])
    matnr = int(row["matnr"])
    try:
        seas_pres = ljung_box_test(df, int(row["matnr"]))[0]
        if not seas_pres:
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
            plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_05/"
                        + str(kunag) + "_" + str(matnr) + "_" + str(dtw_flag) + ".png")
            report = report.append([[kunag, matnr, seas_pres, dtw_flag, seas_error, norm_error]])
    except:
        pass
    count += 1
    print(count)
report.columns = ["kunag", "matnr", "seasonality", "dtw_flag", "seas_error", "norm_error"]
report.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_05/report_2018_02_05.csv")
    # except:
    #     pass