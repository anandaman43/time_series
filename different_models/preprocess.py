import pandas as pd
import datetime


def remove_negative_rows(input_df):
    return input_df[input_df["quantity"] >= 0]


def individual_series(input_df, kunag, matnr):
    """
    selects a dataframe corresponding to a particular kunag and matnr
    param: a pandas dataframe
    return: a pandas dataframe
    """
    df = input_df.copy()
    output_df = df[(df["kunag"] == kunag) & (df["matnr"] == matnr)]
    return output_df


def splitter(input_df):
    """
    splits the data into train and test where test is last 6 months data
    param: pandas dataframe
    returns: two pandas dataframes (train and test)
    """
    df = input_df.copy()
    df["date"] = df["date"].map(str)
    df["parse_date"] = df["date"].apply(lambda dates: pd.datetime.strptime(dates, '%Y%m%d'))
    latest_date = df.sort_values(["parse_date"]).iloc[-1]["parse_date"]
    start_date = df.sort_values(["parse_date"]).iloc[0]["parse_date"]
    validation_date = datetime.datetime(latest_date.year, latest_date.month, latest_date.day) - datetime.timedelta(
        days=180)
    validation = df[df["parse_date"]>=validation_date].drop(["parse_date"], axis=1)
    train = df[df["parse_date"]<validation_date].drop(["parse_date"], axis=1)
    return train, validation
