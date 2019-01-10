import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error
from aggregate import sample_aggregate_seas_5_point


def arima_mse(train, validation, order=(0, 1, 1)):
    train["prediction"] = train["quantity"]
    k = 0
    for index, row in validation.iterrows():
        train["quantity"] = train["quantity"].map(float)
        model1 = sm.tsa.statespace.SARIMAX(train["quantity"], order=order, seasonal_order=(0, 0, 0, 52),
                                           enforce_stationarity=True, enforce_invertibility=True,
                                           measurement_error=False, time_varying_regression=False, mle_regression=True)
        res1 = model1.fit(disp=False)
        predicted = res1.forecast(1)
        row["prediction"] = predicted.values[0]
        train = pd.concat([train, pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            val_index = train.shape[0] - 1
            k = 1
    output_df = train
    test_df = train.iloc[val_index:]
    return mean_squared_error(test_df["quantity"], test_df["prediction"]), output_df


def arima(train, validation, test):

    mse1, output_1 = arima_mse(train, validation, (0, 1, 1))
    mse2, output_2 = arima_mse(train, validation, (0, 2, 2))
    output_val = output_1
    if mse1 <= mse2:
        order = (0, 1, 1)
        mse_val = mse1
    else:
        order = (0, 2, 2)
        output_val = output_2
        mse_val = mse2
    train = pd.concat([train, validation])
    mse, output_df= arima_mse(train, test, order)
    return mse, output_df, output_val, order, mse_val


def arima_mse_seasonality_added(train, validation, order=(0, 1, 1)):
    total_seasonality = sample_aggregate_seas_5_point()
    k = 0
    train["prediction"] = train["quantity"]
    for index, row in validation.iterrows():
        train["quantity"] = train["quantity"].map(float)
        exog_train = total_seasonality.loc[train.set_index("dt_week").index].reset_index(drop=True)
        exog_train = sm.add_constant(exog_train)
        seasonality_pred_index = validation.loc[index]["dt_week"]
        exog_prediction = [[1, total_seasonality.loc[seasonality_pred_index]]]
        model1 = sm.tsa.statespace.SARIMAX(train["quantity"], exog_train, order=order, seasonal_order=(0, 0, 0, 52),
                                           enforce_stationarity=True, enforce_invertibility=True,
                                           measurement_error=False, time_varying_regression=False,
                                           mle_regression=True)
        res1 = model1.fit(disp=False)
        predicted = res1.forecast(1, exog=exog_prediction)
        row["prediction"] = predicted.values[0]
        train = pd.concat([train, pd.DataFrame(row).T]).reset_index(drop=True)
        if k == 0:
            val_index = train.shape[0] - 1
            k = 1
    output_df = train
    test_df = train.iloc[val_index:]
    return mean_squared_error(test_df["quantity"], test_df["prediction"]), output_df


def arima_seasonality_added(train, validation, test):

    mse1, output_1 = arima_mse_seasonality_added(train, validation, (0, 1, 1))
    mse2, output_2 = arima_mse_seasonality_added(train, validation, (0, 2, 2))
    output_val = output_1
    if mse1 <= mse2:
        order = (0, 1, 1)
        mse_val = mse1
    else:
        order = (0, 2, 2)
        output_val = output_2
        mse_val = mse2

    train = pd.concat([train, validation])
    mse, output_df = arima_mse_seasonality_added(train, test, order)
    return mse, output_df, output_val, order, mse_val


if __name__=="__main__":
    arima(train, validation, test)