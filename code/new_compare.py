import pandas as pd
from seasonality_detection import ljung_box_test
from selection import load_data

df = load_data()
# data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/bucket_1_sample_results_5_point.csv")
# data["seasonality"] = 0
# k = 0
# for i in range(data.shape[0]):
#     data["seasonality"].iloc[i] = ljung_box_test(df, int(data["matnr"].iloc[i]))
#     k+=1
#     print(k)
# data.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/bucket_1_sample_results_5_point_check.csv")

data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/bucket_1_sample_results_5_point_check.csv")
data = data[data["seasonality"] == True]
print(data.shape)
data = data[data["score1"] > data["score2"]]
print(data.shape)