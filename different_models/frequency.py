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
    if len_in_days <= 547:
        return -2
    latest_year = latest_date.year-1
    latest_month = latest_date.month
    latest_day = latest_date.day
    last_year_date = datetime(latest_year, latest_month, latest_day)
    freq = input_df[(input_df["parse_date"] >= last_year_date) & (input_df["parse_date"] <= latest_date)].shape[0]
    return freq


if __name__ == "__main__":
    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv",
                     sep="\t")
    print()
