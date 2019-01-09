import pandas as pd
from transformation import remove_negative
from datetime import datetime


def normalized_frequency(input_df):
    """ returns number of datapoints in last year with normalization
        input: a ts with a particular kunag and matnr
        output: an integer
    """
    input_df = remove_negative(input_df)
    input_df["parse_date"] = pd.to_datetime(input_df["date"], format="%Y%m%d")
    input_df = input_df.sort_values("parse_date")
    if input_df.shape[0] <= 5:
        return 0
    latest_date = input_df["parse_date"].iloc[-1]
    first_date = input_df["parse_date"].iloc[0]
    len_in_days = (latest_date - first_date).days
    if len_in_days <= 365:
        return -1
    if len_in_days <= 730:
        return -2
    latest_year = latest_date.year-1
    latest_month = latest_date.month
    latest_day = latest_date.day
    last_year_date = datetime(latest_year, latest_month, latest_day)
    freq = input_df[(input_df["parse_date"] >= last_year_date) & (input_df["parse_date"] <= latest_date)].shape[0]
    return freq


def normalized_frequency_and_days(input_df, kunag, matnr):
    df_copy = input_df.copy()
    df_copy = remove_negative(df_copy)
    df_copy = df_copy[(df_copy["kunag"] == kunag) & (df_copy["matnr"] == matnr)]
    df_copy["parse_date"] = pd.to_datetime(df_copy["date"], format="%Y%m%d")
    df_copy = df_copy.sort_values("parse_date")
    if df_copy.shape[0] == 0:
        return 0, 0
    end_date = df_copy["parse_date"].iloc[-1]
    start_date = df_copy["parse_date"].iloc[0]
    num_days = (end_date - start_date).days
    last_year_date = datetime(end_date.year-1, end_date.month, end_date.day)
    if num_days == 0:
        return 0, 0
    if num_days > 365:
        freq = df_copy[(df_copy["parse_date"] >= last_year_date) & (df_copy["parse_date"] <= end_date)].shape[0]
        return freq, num_days
    else:
        freq = df_copy[(df_copy["parse_date"] >= last_year_date) & (df_copy["parse_date"] <= end_date)].shape[0]
        norm_freq = (365/num_days)*freq
        return norm_freq, num_days


if __name__ == "__main__":
    from tqdm import tqdm
    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv",
                     sep="\t")
    frequency_days_cleaveland = pd.DataFrame()
    for index, group in tqdm(df.groupby(["kunag", "matnr"])):
        freq, days = normalized_frequency_and_days(df, index[0], index[1])
        frequency_days_cleaveland = frequency_days_cleaveland.append([[index[0], index[1], freq, days]])
    frequency_days_cleaveland.columns = ["kunag", "matnr", "frequency", "days"]
    frequency_days_cleaveland.to_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_cleaveland.csv")
