import pandas as pd
from selection import load_data
from seasonality_detection import ljung_box_test
from dtw_check import dtw_check
from hypothesis_2 import arima_seasonality_added_rolling
from hypothesis_2 import arima_rolling_011
from hypothesis_2 import arima_seasonality_added_rolling_022
from hypothesis_2 import arima_rolling
from selection import individual_series
from stl_decompose import product_seasonal_comp_7_point
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import os


project = "/home/aman/PycharmProjects/seasonality_hypothesis/"
folder = "category_7_seasonality"
file = "seasonality.csv"
folder_address = os.path.join(project, folder)
file_address = os.path.join(folder_address, file)
# os.mkdir(folder_address)

df = load_data()
frequency_cleaveland = pd.read_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
# bucket_1_sample = frequency_cleaveland[(frequency_cleaveland["frequency"] > 26) & (frequency_cleaveland["days"] > 730)].sample(400, random_state=1)
# bucket_1_sample.to_csv(
#     "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv", index=False)
sample = frequency_cleaveland[(frequency_cleaveland["frequency"] >= 26) & (frequency_cleaveland["days"] > 92) &
                              (frequency_cleaveland["days"] <= 365 + 183)]
sample = sample["matnr"].unique()
# sample.to_csv(folder_address+"/sample.csv")
# sample = pd.read_csv(folder_address+"/sample.csv")
report = pd.DataFrame()
count = 0
# start = time.time()
for matnr in sample:
        if count<321:
                count +=2
                continue
        test1 = ljung_box_test(df, matnr)
        test_flag = test1[0]
        test_p_value = test1[1]
        aggregated_data = test1[4]
        plt.figure(figsize=(16, 8))
        plt.plot(aggregated_data.set_index("dt_week")["quantity"], marker=".", markerfacecolor="red", label="aggregated_data")
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Quantity", fontsize=14)
        plt.legend()
        plt.savefig(folder_address + "/" + str(test_flag) + "_" + str(test_p_value) + str(matnr) + ".png")
        count += 1
        print("count;", count)
        report = report.append([[matnr, test_flag, test_p_value]])
        report.to_csv(file_address, index=False)
        count += 1
        pass
report.columns = ["matnr", "test_flag", "test_p_value"]
report.to_csv(file_address, index=False)