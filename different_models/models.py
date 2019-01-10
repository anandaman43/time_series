import pandas as pd
import numpy as np
import datetime
import data_transformation
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from preprocess import remove_negative_rows
from preprocess import individual_series
from preprocess import splitter
from fbprophet import Prophet


def naive(input_df, kunag, matnr):
    """
    applies naive model and calculates mse score on test data
    :param input_df:
    :param kunag:
    :param matnr:
    :return:
    """
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x: x.replace("-", ""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        row["prediction"] = df_series_train["quantity"].iloc[-1]
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def average(input_df, kunag, matnr):
    """
    applies average model and calculates mse score on test data
    :param input_df:
    :param kunag:
    :param matnr:
    :return:
    """
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-",""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        row["prediction"] = df_series_train["quantity"].mean()
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def moving_average(input_df, kunag, matnr, q):
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-",""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        row["prediction"] = df_series_train["quantity"].rolling(q).mean().iloc[-1]
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def drift_prediction(input_df):

    df = input_df.copy()
    df = df.reset_index(drop=True)
    y_0 = df.iloc[0]["quantity"]
    y_1 = df.iloc[-1]["quantity"]
    slope = (y_1 - y_0)/float(df.shape[0])
    predicted = df.iloc[-1]["quantity"] + slope
    return predicted


def drift(input_df, kunag, matnr):

    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-",""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        row["prediction"] = drift_prediction(df_series_train)
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def simple_exponential_smoothing(input_df, kunag, matnr, alpha=0.6):

    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-", ""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        fit2 = SimpleExpSmoothing(np.asarray(df_series_train["quantity"])).fit(smoothing_level=alpha,optimized=False)
        row["prediction"] = fit2.forecast(1)
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def holts_linear_trend(input_df, kunag, matnr, smoothing_level=0.3, smoothing_slope=0.1):
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-", ""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        df_series_train["quantity"] = df_series_train["quantity"].map(float)
        fit1 = Holt(np.asarray(df_series_train["quantity"])).fit(smoothing_level=smoothing_level,
                                                                 smoothing_slope=smoothing_slope)
        predicted = fit1.forecast(1)
        row["prediction"] = predicted[0]
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def holts_winter_method(input_df, kunag, matnr, seasonal_period, alpha=None, beta=None, gamma=None):
    """

    :param input_df:
    :param kunag:
    :param matnr:
    :param seasonal_period:
    :param alpha:
    :param beta:
    :param gamma:
    :return:
    """
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x:x.replace("-",""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        df_series_train["quantity"] = df_series_train["quantity"].map(float)
        fit1 = ExponentialSmoothing(np.asarray(df_series_train["quantity"]), seasonal_periods=seasonal_period,
                                    trend='add', seasonal='add',).fit(smoothing_level=alpha, smoothing_slope=beta,
                                                                      smoothing_seasonal=gamma)
        predicted = fit1.forecast(1)
        row["prediction"] = predicted[0]
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def output_plot(input_df, mse):
    input_df.Timestamp = pd.to_datetime(input_df.date,format='%Y%m%d') 
    input_df.index = input_df.Timestamp
    plt.figure(figsize=(16,8))
    plt.plot(input_df["prediction"], label='prediction')
    plt.plot(input_df["quantity"], label='actual')
    plt.title("mse error is : " + str(mse))
    plt.show()


def arima(input_df, kunag, matnr, p=2, d=1, q=4, trend="n"):
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr)
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x: x.replace("-", ""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index,row in df_series_test.iterrows():
        df_series_train["quantity"] = df_series_train["quantity"].map(float)
        fit1 = sm.tsa.statespace.SARIMAX(df_series_train["quantity"], order=(p, d, q), trend=trend).fit()
        predicted = fit1.forecast(1)
        row["prediction"] = predicted.values[0]
        df_series_train = pd.concat([df_series_train,pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    # print("mean squared error is :",mean_squared_error(output_df["quantity"], output_df["prediction"]))
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


def prophet_preprocess(ts):
    ts["ds"] = ts["dt_week"]
    ts["y"] = ts["quantity"]
    output_df = ts[["ds", "y"]].sort_values("ds")
    return output_df


def prophet(input_df, kunag, matnr, growth="linear", changepoint_prior_scale=0.5, yearly_seasonality=False,
            daily_seasonality=False, weekly_seasonality=False, seasonality_prior_scale=100):
    df = input_df.copy()
    df = remove_negative_rows(df)
    df_series = individual_series(df, kunag, matnr, )
    df_series = data_transformation.get_weekly_aggregate(df_series)
    df_series["date"] = df_series["dt_week"].map(str)
    df_series["date"] = df_series["date"].apply(lambda x: x.replace("-", ""))
    df_series["prediction"] = df_series["quantity"]
    df_series_train, df_series_test = splitter(df_series)
    k = 0
    for index, row in df_series_test.iterrows():
        df_series_train["quantity"] = df_series_train["quantity"].map(float)
        df_copy = df_series_train.copy()
        ts = prophet_preprocess(df_copy)
        m = Prophet(growth=growth, changepoint_prior_scale=changepoint_prior_scale,
                    yearly_seasonality=yearly_seasonality, daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality, seasonality_prior_scale=seasonality_prior_scale)
        m.fit(ts)
        future = m.make_future_dataframe(periods=1, freq="W", include_history=False).apply(
            lambda x: (x + pd.Timedelta(4, unit="D")))  # + pd.Timedelta(4, unit="D")

        forecast = m.predict(future)
        row["prediction"] = forecast.iloc[0]["yhat"]
        df_series_train = pd.concat([df_series_train, pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            test_index = df_series_train.shape[0] - 1
            k = 1
    output_df = df_series_train
    test_df = df_series_train.iloc[test_index:]
    return output_df, mean_squared_error(test_df["quantity"], test_df["prediction"])


if __name__== "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv", sep = "\t")

    # output, mse = naive(df, 500057582, 103029)
    # output, mse = average(df, 500057582, 103029)
    # output, mse = drift(df, 500057582, 152222)
    # output, mse = simple_exponential_smoothing(df, 500057582, 103029, 0.1)
    # output, mse = holts_linear_trend(df, 500057582, 103029, 0.9, 0.1)
    # output, mse = holts_winter_method(df, 500057582, 103029, 4, 0.5, 0.5, 0.1)
    # output, mse = arima(df, 500057582, 103029, 3, 4, 4, "n")
    # output, mse = moving_average(df, 500057582, 117803, 24)
    output, mse = prophet(df, 500057582, 103029)
    # output.to_csv("/home/aman/Desktop/CSO_drug/file_generated/output.csv")
    # print(output)
    output_plot(output, mse)