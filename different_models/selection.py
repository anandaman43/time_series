import pandas as pd
from transformation import remove_negative
from frequency import normalized_frequency
import models
pd.set_option('display.max_columns', 10)
import matplotlib.pyplot as plt


def select_series(input_df, kunag=500057582, matnr=103029):
    """ selects a series corresponding to the given kunag and matnr """
    output_ts = pd.DataFrame()
    output_ts = input_df[(input_df["kunag"] == kunag) & (input_df["matnr"] == matnr)]
    output_ts = remove_negative(output_ts)
    return output_ts


def load_data():
    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv",
                     sep="\t")
    return df


if __name__ == "__main__":
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore")
    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv",
                     sep="\t")
    # frequency_cleaveland = pd.DataFrame()
    # for index, group in df.groupby(["kunag", "matnr"]):
    #     ts = select_series(df, index[0], index[1])
    #     freq = normalized_frequency(ts)
    #     frequency_cleaveland = frequency_cleaveland.append([[index[0], index[1], freq]])
    # frequency_cleaveland.columns = ["kunag", "matnr", "frequency"]
    # frequency_cleaveland.to_csv("/home/aman/Desktop/CSO_drug/file_generated/frequency_cleaveland.csv")
    frequency_cleaveland = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/frequency_cleaveland.csv")
    bucket_1_sample = frequency_cleaveland[frequency_cleaveland["frequency"] > 26].sample(200, random_state=1)
    # mse = 0
    # for index, row in bucket_1_sample.iterrows():
    #     mse += models.arima(df, row["kunag"], row["matnr"], 3, 0, 0)[1]
    # print(mse)
    import itertools
    import time

    p = [0]
    d = [1, 2]
    q = [1, 2]
    results = pd.DataFrame()
    for parameters in tqdm(itertools.product(p, d, q)):
        try:
            mse = 0
            start = time.time()
            for index, row in bucket_1_sample.iterrows():

                mse += models.arima(df, row["kunag"], row["matnr"], parameters[0], parameters[1],
                                                     parameters[2])[1]
            end = time.time()
            results = results.append([[parameters[0], parameters[1], parameters[2], mse]])
            print("p:", parameters[0], "d:", parameters[1], "q:", parameters[2],
                  "mse:", mse, "time:", (end-start)/60)
            # results.columns = ["p", "d", "q", "mse"]
            print("saved1")
            results.to_csv("/home/aman/Desktop/CSO_drug/file_generated/arima_results.csv")
            print("saved2")
        except:
            pass

