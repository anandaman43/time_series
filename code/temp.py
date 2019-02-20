# from outlier import ma_replace_outlier
# from selection import load_data
# from selection import remove_negative_rows
# from data_transformation import get_weekly_aggregate
# from outlier import ma_replace_outlier
# from dateutil import parser
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

import pandas as pd
d = {"A": [1,1,3], "B" : [4,5,6]}
df = pd.DataFrame(d)
print(df)
def printa(x):
    print(x)
print(df.groupby(["A"]).agg(lambda x:printa(x)))