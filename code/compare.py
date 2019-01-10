import pandas as pd
from selection import load_data
from selection import individual_series
from hypothesis import arima
from hypothesis import arima_seasonality_added
from preprocess import splitter_2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
result = pd.DataFrame()
count1 = 0
count2 = 0
error = 0
for index, row in tqdm(sample.iterrows()):
    df_series = individual_series(df, row["kunag"], row["matnr"])
    train, validation, test = splitter_2(df_series)
    try:
        score1 = arima(train, validation, test)[0]
        score2 = arima_seasonality_added(train, validation, test)[0]
        result = result.append([[row["kunag"], row["matnr"], score1, score2]])
        if score1 < score2:
            count1 += 1
        elif score1 >= score2:
            count2 += 1
    except:
        error += 1
        print("kunag:", row["kunag"], "matnr:", row["matnr"])
    print("count1 :", count1, "count2 :", count2, "error :", error)
print("count1 :", count1, "count2 :", count2, "error :", error)
result.columns = ["kunag", "matnr", "score1", "score2"]
result.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample_results_5_point_sample.csv")
print(result)
