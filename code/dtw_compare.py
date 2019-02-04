import pandas as pd

# data1 = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_01/report_2018_02_01.csv")
# data2 = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/dtw_2018_02_04/result.csv")
# data3 = data1.copy()
# data3["d"] = 0
# for i in range(len(data1)):
#     matnr = data1.iloc[i]["matnr"]
#     kunag = data1.iloc[i]["kunag"]
#     data3["d"].iloc[i] = data2[(data2["matnr"] == matnr) & (data2["kunag"] == kunag)]["d"].values[0]
# data3.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/dtw_2018_02_04/result_dtw.csv")

data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/dtw_2018_02_04/result_dtw.csv")
data = data.sort_values("d")


def sigmoid(s):
    if s>=0:
        return 1
    else:
        return 0


data["diff"] = data["diff"].apply(lambda x:sigmoid(x))
threshold = 0.009
correct = data[data["d"] <= threshold]["diff"].value_counts().loc[1]
wrong = data[data["d"] <= threshold]["diff"].value_counts().loc[0]
print("Total: ", (correct+wrong))
print("correct: ", (correct))
print("percentage: ", correct*100/(correct+wrong))
# print(data)