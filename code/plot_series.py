import pandas as pd
from selection import load_data
from selection import individual_series
from hypothesis import arima
from hypothesis import arima_seasonality_added
from preprocess import splitter_2
from tqdm import tqdm
import matplotlib.pyplot as plt
from seasonality import product_seasonal_comp_5_point
from seasonality_detection import ljung_box_test
import warnings
warnings.filterwarnings("ignore")


def function1(df, kunag, matnr):
    df_series = individual_series(df, kunag, matnr)
    train, validation, test = splitter_2(df_series)
    seas_pres = ljung_box_test(df, matnr)
    if not seas_pres:
        return None
    seasonality_product = product_seasonal_comp_5_point(df, matnr)
    score1, output1, output1_val, order1, mse_val_1 = arima_seasonality_added(train, validation, test, seasonality_product)
    score2, output2, output2_val, order2, mse_val_2 = arima(train, validation, test)
    input_df1 = output1.set_index("dt_week")
    input_df2 = output2.set_index("dt_week")
    output1_val = output1_val.set_index("dt_week")
    output2_val = output2_val.set_index("dt_week")
    plt.figure(figsize=(16, 8))
    plt.plot(input_df1["prediction"], marker=".", color='red', label='arima_with_seasonality')
    plt.plot(input_df2["prediction"], marker=".", color='blue', label='arima')
    plt.plot(output1_val["prediction"], marker=".", color='brown', label='arima_seasonality_validation')
    plt.plot(output2_val["prediction"], marker=".", label='arima_validation')
    plt.plot(input_df1["quantity"], marker=".", color='orange', label='actual')
    plt.xlabel('time in weeks')
    plt.ylabel('quantities')
    if score2 < score1:
        plt.title('normal')
    else:
        plt.title('seasonality')
    plt.text("05-04-2018", 0.1, 'seasonality = ' + str(order1) + '\n'
                                 'test_mse_seasonality = ' + str(score1) + '\n'  
                                 'validation_mse_seasonality = ' + str(mse_val_1) + '\n'                                          
                                 'normal = ' + str(order2) + '\n'
                                 'test_mse_normal = ' + str(score2) + '\n'
                                 'validation_mse_normal = ' + str(mse_val_2) + '\n')
    plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/plots_seasonality_108/" + str(kunag) + "_" + str(matnr) + ".png")
    #plt.show()


df = load_data()
sample = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/bucket_1_sample.csv")
for index, row in tqdm(sample.iterrows()):
    function1(df, row["kunag"], row["matnr"])

