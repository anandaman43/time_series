import statsmodels.api as sm
import pandas as pd
from sklearn.metrics import mean_squared_error
#from aggregate import overall_aggregate_seas_5_point
from hypothesis import arima_mse
from hypothesis import arima
from hypothesis import arima_mse_seasonality_added


def splitter_moving(input_df):
    """
    splits the data into train and test where test is last 6 months data
    param: pandas dataframe
    returns: two pandas dataframes (train and test)
    """
    df = input_df.copy()
    df = df.sort_values("dt_week")
    test = df.iloc[-1:]
    # print("test:",test.shape)
    validation = df.iloc[-25:-1]
    # print("validation::",validation.shape)
    train = df.iloc[0:-25]
    # print("train:",train.shape)
    return train, validation, test


def arima_seasonality_added_rolling(input_df, seasonality):
    for i in range(15, -1, -1):
        if i > 0:
            train_copy = input_df.copy().iloc[0:-i]
        else:
            train_copy = input_df.copy().iloc[0:]
        train, validation, test = splitter_moving(train_copy)
        # print(train)
        # print(validation)
        # print(test)
        mse1, output_1 = arima_mse_seasonality_added(train, validation, seasonality, order=(0, 1, 1))
        mse2, output_2 = arima_mse_seasonality_added(train, validation, seasonality, order=(0, 2, 2))
        mse3, output_3 = arima_mse(train, validation, (0, 1, 1))
        mse4, output_4 = arima_mse(train, validation, (0, 2, 2))
        if mse1 <= mse2:
            order = (0, 1, 1)
            mse_val = mse1
            seas = True
        else:
            order = (0, 2, 2)
            output_val = output_2
            mse_val = mse2
            seas = True
        if mse_val > mse3:
            order = (0, 1, 1)
            mse_val = mse3
            output_val = output_3
            seas = False
        elif mse_val > mse4 and mse4 < mse3:
            order = (0, 2, 2)
            output_val = output_4
            mse_val = mse2
            seas = False
        if seas:
            mse1, output_1 = arima_mse_seasonality_added(pd.concat([train, validation]), test, seasonality,
                                                         order=(0, 1, 1))
        else:
            mse1, output_1 = arima_mse(pd.concat([train, validation]), test, order)
        if i == 15:
            output_test = pd.concat([train, validation])
            output_test["prediction"] = output_test["quantity"]
        output_test = pd.concat([output_test, pd.DataFrame(output_1.iloc[-1]).T])
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