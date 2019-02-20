import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error
#from aggregate import overall_aggregate_seas_5_point
from hypothesis import arima_mse
from hypothesis import arima
from hypothesis import arima_mse_seasonality_added


def splitter_moving(input_df):
    """
    splits the data into train and test where test is last point and validation is last 24 points before test
    param: pandas dataframe : input_df
    returns: three pandas dataframes (train, validation and test)
    """
    df = input_df.copy()
    df = df.sort_values("dt_week")
    test = df.iloc[-1:]
    validation = df.iloc[-25:-1]
    train = df.iloc[0:-25]
    return train, validation, test


def arima_rolling(input_df):
    for i in range(15, -1, -1):
        if i > 0:
            train_copy = input_df.copy().iloc[0:-i]
        else:
            train_copy = input_df.copy().iloc[0:]
        train, validation, test = splitter_moving(train_copy)
        mse1, output_1 = arima_mse(train, validation, (0, 1, 1))
        mse2, output_2 = arima_mse(train, validation, (0, 2, 2))
        mse3, output_3 = arima_mse(train, validation, (0, 1, 2))
        if (mse1 <= mse2) & (mse1 <= mse3):
            order = (0, 1, 1)
        elif mse2 <= mse3:
            order = (0, 2, 2)
        else:
            order = (0, 1, 2)
        mse, output = arima_mse(pd.concat([train, validation]), test, order)
        if i == 15:
            output_test = pd.concat([train, validation])
            output_test["prediction"] = output_test["quantity"]
        output_test = pd.concat([output_test, pd.DataFrame(output.iloc[-1]).T])
    return output_test


def arima_seasonality_added_rolling(input_df, seasonality):
    for i in range(15, -1, -1):
        if i > 0:
            train_copy = input_df.copy().iloc[0:-i]
        else:
            train_copy = input_df.copy().iloc[0:]
        train, validation, test = splitter_moving(train_copy)
        # order = (0, 1, 1)
        mse1, output_1 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 1))
        mse2, output_2 = arima_mse_seasonality_added(train, validation, seasonality, (0, 2, 2))
        mse3, output_3 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 2))
        if (mse1 <= mse2) & (mse1 <= mse3):
            order = (0, 1, 1)
        elif mse2 <= mse3:
            order = (0, 2, 2)
        else:
            order = (0, 1, 2)
        mse, output = arima_mse_seasonality_added(pd.concat([train, validation]), test, seasonality,  order)
        if i == 15:
            output_test = pd.concat([train, validation])
            output_test["prediction"] = output_test["quantity"]
        output_test = pd.concat([output_test, pd.DataFrame(output.iloc[-1]).T])
    return output_test


def arima_seasonality_added_rolling_011(input_df, seasonality):
    for i in range(15, -1, -1):
        if i > 0:
            train_copy = input_df.copy().iloc[0:-i]
        else:
            train_copy = input_df.copy().iloc[0:]
        train, validation, test = splitter_moving(train_copy)
        order = (0, 1, 1)
        # mse1, output_1 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 1))
        # mse2, output_2 = arima_mse_seasonality_added(train, validation, seasonality, (0, 2, 2))
        # mse3, output_3 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 2))
        # if (mse1 <= mse2) & (mse1 <= mse3):
        #     order = (0, 1, 1)
        # elif mse2 <= mse3:
        #     order = (0, 2, 2)
        # else:
        #     order = (0, 1, 2)
        mse, output = arima_mse_seasonality_added(pd.concat([train, validation]), test, seasonality, order)
        if i == 15:
            output_test = pd.concat([train, validation])
            output_test["prediction"] = output_test["quantity"]
        output_test = pd.concat([output_test, pd.DataFrame(output.iloc[-1]).T])
    return output_test


def arima_seasonality_added_rolling_022(input_df, seasonality):
    for i in range(15, -1, -1):
        if i > 0:
            train_copy = input_df.copy().iloc[0:-i]
        else:
            train_copy = input_df.copy().iloc[0:]
        train, validation, test = splitter_moving(train_copy)
        order = (0, 2, 2)
        # mse1, output_1 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 1))
        # mse2, output_2 = arima_mse_seasonality_added(train, validation, seasonality, (0, 2, 2))
        # mse3, output_3 = arima_mse_seasonality_added(train, validation, seasonality, (0, 1, 2))
        # if (mse1 <= mse2) & (mse1 <= mse3):
        #     order = (0, 1, 1)
        # elif mse2 <= mse3:
        #     order = (0, 2, 2)
        # else:
        #     order = (0, 1, 2)
        mse, output = arima_mse_seasonality_added(pd.concat([train, validation]), test, seasonality, order)
        if i == 15:
            output_test = pd.concat([train, validation])
            output_test["prediction"] = output_test["quantity"]
        output_test = pd.concat([output_test, pd.DataFrame(output.iloc[-1]).T])
    return output_test


if __name__=="__main__":
    from selection import load_data
    from selection import individual_series
    from preprocess import splitter_2
    from stl_decompose import product_seasonal_comp_7_point
    import matplotlib.pyplot as plt
    df = load_data()
    sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
    count = 0
    for index, row in sample.iterrows():
        count += 1
        try:
            kunag = int(row["kunag"])
            matnr = int(row["matnr"])
            # kunag = 500068486
            # matnr = 134926
            df_series = individual_series(df, kunag, matnr)
            seasonality_product = product_seasonal_comp_7_point(df, matnr)
            result = arima_seasonality_added_rolling(df_series, seasonality_product)
            result = result.set_index("dt_week")
            new_error = mean_squared_error(result.iloc[-16:]["quantity"], result.iloc[-16:]["prediction"])
            #print(result.iloc[-16:])
            plt.figure(figsize=(16, 8))
            plt.plot(result["quantity"], marker=".")
            plt.plot(result["prediction"], marker=".")
            train, validation, test = splitter_2(df_series)
            score2, output2, output2_val, order2, mse_val_2 = arima(train, validation, test)
            input_df2 = output2.set_index("dt_week")
            output2_val = output2_val.set_index("dt_week")
            test_norm = pd.concat([output2_val, input_df2.iloc[-16:]])
            old_error = mean_squared_error(input_df2.iloc[-16:]["quantity"], input_df2.iloc[-16:]["prediction"])
            #print(input_df2.iloc[-16:])
            plt.plot(test_norm["prediction"], marker=".", color='blue', label='test_normal')
            plt.plot(output2_val["prediction"], marker=".", label='val_normal')
            plt.title("new_error = "+str(new_error) + "   old_error = " + str(old_error))
            plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/latest_plots/"+str(matnr)+"_"+str(kunag))
            # plt.show()
            print(str(count) + "done")
        except:
            pass