import pandas as pd
from selection import load_data
from selection import individual_series
from hypothesis import arima
from hypothesis import arima_seasonality_added
from preprocess import splitter_2
from tqdm import tqdm
from stl_decompose import product_seasonal_comp_7_point
from selection import load_data
import warnings
from seasonality_detection import ljung_box_test
warnings.filterwarnings("ignore")


df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
result = pd.DataFrame()
count1 = 0
count2 = 0
error = 0
for index, row in sample.iterrows():
    print("kunag: ", row["kunag"], " matnr: ", row["matnr"])
    seas_pres = ljung_box_test(df, row["matnr"])
    print("Seasonality :", seas_pres)
    df_series = individual_series(df, row["kunag"], row["matnr"])
    train, validation, test = splitter_2(df_series)
    if not seas_pres:
        continue
    seasonality_product = product_seasonal_comp_7_point(df, int(row["matnr"]))
    score1 = arima(train, validation, test)
    score1 = score1[0] + score1[4]
    print("score1=", score1)
    score2 = arima_seasonality_added(train, validation, test, seasonality_product)
    score2 = score2[0] + score2[4]
    print("score2=", score2)
    result = result.append([[row["kunag"], row["matnr"], score1, score2]])
    if score1 < score2:
        count1 += 1
    elif score1 >= score2:
        count2 += 1
    print("count1 :", count1, "count2 :", count2, "error :", error)
result.columns = ["kunag", "matnr", "score1", "score2"]
result.to_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample_results_7_point_seasonality_18th_jan_thresgh_0.01_stl_test_val.csv")
print(result)
