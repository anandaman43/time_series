from selection import load_data
from selection import remove_negative_rows
import pandas as pd
from preprocess import splitter_2
from hypothesis import arima
from selection import individual_series


def individual_series_2(input_df, kunag=500057582, matnr=103029):
    """
    selects a dataframe corresponding to a particular kunag and matnr
    param: a pandas dataframe
    return: a pandas dataframe
    """
    df_copy = input_df.copy()
    df_copy = remove_negative_rows(df_copy)
    df_copy = df_copy[df_copy["date"] >= 20160703]
    output_df = df_copy[(df_copy["kunag"] == kunag) & (df_copy["matnr"] == matnr)]
    output_df["dt_week"] = output_df["date"].apply(lambda x: pd.to_datetime(x, format="%Y%m%d"))
    output_df = output_df.sort_values("dt_week")
    output_df = output_df.set_index("dt_week")
    return output_df


if __name__ == "__main__":
    print(individual_series_2(load_data()))
    df_series = individual_series(load_data(), 500057582, 103029)
    train, validation, test = splitter_2(df_series)
    print(arima(train, validation, test)[1])