import pandas as pd
from data_transformation import get_weekly_aggregate
import matplotlib.pyplot as plt
from dateutil import parser
from outlier import ma_replace_outlier
import statsmodels.api as sm
pd.set_option('display.max_columns', 10)


# def load_data():
#     df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv",
#                      sep="\t")
#     return df
def load_data():
    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/4200_C005_raw_invoices_2019-01-06.tsv",
                     names=["kunag", "matnr", "date", "quantity", "price"])
    return df


def remove_negative_rows(input_df):
    return input_df[input_df["quantity"] >= 0]


def individual_series(input_df, kunag=500057582, matnr=103029):
    """
    selects a dataframe corresponding to a particular kunag and matnr
    param: a pandas dataframe
    return: a pandas dataframe
    """
    df_copy = input_df.copy()
    df_copy = remove_negative_rows(df_copy)
    df_copy = df_copy[df_copy["date"] >= 20160703]
    output_df = df_copy[(df_copy["kunag"] == kunag) & (df_copy["matnr"] == matnr)]
    output_df = get_weekly_aggregate(output_df)
    output_df["dt_week"] = output_df["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    outlier_removed = outlier(output_df)
    return outlier_removed


def plot_series(df_series):
    df_series.Timestamp = pd.to_datetime(df_series.dt_week, format='%Y-%m-%d')
    df_series.index = df_series.Timestamp
    plt.figure(figsize=(12, 8))
    plt.plot(df_series["quantity"])
    plt.show()


def select_series(input_df, kunag=500057582, matnr=103029):
    """ selects a series corresponding to the given kunag and matnr """
    output_ts = pd.DataFrame()
    output_ts = input_df[(input_df["kunag"] == kunag) & (input_df["matnr"] == matnr)]
    output_ts = remove_negative_rows(output_ts)
    return output_ts


def validation_preprocess(df_series):
    df_series["prediction"] = df_series["quantity"]
    return df_series


def validation_mse(train, validation, order=(0, 1, 1)):
    train = validation_preprocess(train)
    validation = validation_preprocess(validation)
    k = 0
    for index, row in validation.iterrows():
        train["quantity"] = train["quantity"].map(float)
        model1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order)
        res1 = model1.fit(disp=False)
        print(train)
        print(res1.forecast(1))
        break
        row["prediction"] = predicted.values[0]
        train = pd.concat([train, pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = train.shape[0] - 1
            k = 1
    output_df = train
    test_df = train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    #return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def outlier(df_series):
    _testing = df_series[["quantity", "dt_week"]].copy()
    aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

    aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
    aggregated_data.y = aggregated_data.y.apply(float)
    aggregated_data = aggregated_data.sort_values('ds')
    aggregated_data = aggregated_data.reset_index(drop=True)

    _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
    result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
    return result


if __name__ == "__main__":
    from matplotlib import pyplot
    from statsmodels.tsa.seasonal import seasonal_decompose
    import statsmodels.api as sm
    df = load_data()
    df = remove_negative_rows(df)
    df_series = individual_series(df)
    print(df_series.head())
    print(validation_preprocess(df_series.head()))

