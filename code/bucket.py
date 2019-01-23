from selection import load_data
from selection import select_series
from frequency import normalized_frequency
import pandas as pd


def bucket():
    df = load_data()
    frequency_cleaveland = pd.DataFrame()
    for index, group in df.groupby(["kunag", "matnr"]):
        ts = select_series(df, index[0], index[1])
        freq = normalized_frequency(ts)
        frequency_cleaveland = frequency_cleaveland.append([[index[0], index[1], freq]])
    frequency_cleaveland.columns = ["kunag", "matnr", "frequency"]
    frequency_cleaveland.to_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_cleaveland.csv", index=False)


if __name__ == "__main__":
    # bucket()
    # print("done")
    frequency_cleaveland = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
    bucket_1_sample = frequency_cleaveland[(frequency_cleaveland["frequency"] > 26) & (frequency_cleaveland["days"] > 730)].sample(200, random_state=1)
    bucket_1_sample.to_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv", index=False)
    print("done")
