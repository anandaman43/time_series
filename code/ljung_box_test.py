from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from tqdm import tqdm
from data_transformation import get_weekly_aggregate
from dateutil import parser
from outlier import ma_replace_outlier
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


def cond_check(final_df, period=4):
    length = final_df.shape[0]
    if length < 112:
        return None, False
    else:
        extra = int(length % period)
        return final_df.iloc[extra:length], True


def monthly_aggregate(detrended_df, period=4):
    input_df = detrended_df.copy()
    length = input_df.shape[0]
    input_df = input_df.reset_index()
    output_df = input_df.copy()
    for i in range(0, length, period):
        output_df["quantity"].loc[i] = input_df.iloc[i:i+period]["quantity"].sum()
        output_df = output_df.drop(list(range(i + 1, i + period)), axis=0)
    output_df = output_df.set_index("dt_week")
    return output_df


def detrend(input_df, order=18):
    input_df_copy = input_df.copy()
    input_df_copy = input_df_copy.reset_index()
    length = input_df_copy.shape[0]
    output_df = input_df_copy.copy()
    for i in range(order, length-order):
        output_df["quantity"].iloc[i] = input_df_copy.iloc[i-order:i+order+1]['quantity'].mean()
    output_df = output_df.iloc[order:length-order]
    output_df = output_df.set_index("dt_week")
    input_df_copy = input_df_copy.iloc[order:length-order].set_index("dt_week")
    # print(input_df_copy)
    # print(output_df)
    # print(input_df_copy - output_df)
    # plt.plot(output_df, marker=".")
    # plt.title("trend")
    # plt.show()
    detrended = input_df_copy - output_df
    return detrended


def dickey_fuller_test(input_df, matnr=112260):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_cleaveland.csv")
    overall = overall[overall["matnr"] == matnr]
    product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    product_name = product[product["matnr"] == str(matnr)]["description"].values[0]
    k = 0
    for index, row in tqdm(overall.iterrows()):
        frequency = row["frequency"]
        days = row["days"]
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        if frequency == 0:
            continue
        #print(df_series)
        df_series = get_weekly_aggregate(df_series)
        _testing = df_series[["quantity", "dt_week"]].copy()
        aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

        aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
        aggregated_data.y = aggregated_data.y.apply(float)
        aggregated_data = aggregated_data.sort_values('ds')
        aggregated_data = aggregated_data.reset_index(drop=True)
        outlier = True
        if (frequency >= 26) & (days > 365 + 183):
            n_pass = 3
            window_size = 12
            sigma = 4.0
        elif(frequency >= 20) & (frequency < 26):
            n_pass = 3
            window_size = 12
            sigma = 5.0
        elif (frequency >= 26) & (days <= 365+183):
            if len(aggregated_data) >= 26:
                n_pass = 3
                window_size = 12
                sigma = 4.0
            else:
                outlier = False
        elif (frequency >= 12) & (frequency < 20):
            if len(aggregated_data) >= 26:
                n_pass = 3
                window_size = 24
                sigma = 5.0
            else:
                outlier = False
        elif frequency < 12:
            outlier = False

        if outlier:
            _result = ma_replace_outlier(data=aggregated_data, n_pass=n_pass, aggressive=True, window_size=window_size,
                                         sigma=sigma)
            result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        else:
            result = aggregated_data.rename(columns={'ds': 'dt_week', 'y': 'quantity'})
        if k == 1:
            final = pd.concat([final, result])
        if k == 0:
            final = result
            k = 1
    final = final.groupby("dt_week")["quantity"].sum().reset_index()
    final = final.set_index("dt_week")
    #temp = final
    plt.figure(figsize=(16, 8))
    plt.plot(final, marker=".")
    # plt.show()
    final, Flag = cond_check(final)
    if Flag:
        final_detrended = detrend(final)
        # plt.plot(final_detrended, marker=".")
        # plt.title("detrended")
        # plt.show()
        final_aggregate = monthly_aggregate(final_detrended)
        # plt.figure(figsize=(16, 8))
        # plt.plot(final_aggregate, marker=".")
        result = acorr_ljungbox(final_aggregate["quantity"], lags=[13])
        print("statistic: %f" %result[0])
        print("p-value: %f" %result[1])
        plt.title("statistic: " + str(result[0]) + "p-value :" + str(result[1]))
        plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/temp2/" + str(matnr) + "_" + product_name + ".png")
    else:
        print("length of series is less than 112")



if __name__ == "__main__":
    from selection import load_data
    # df = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data/4200_C005_raw_invoices_2019-01-06.tsv",
    #                  names=["kunag", "matnr", "date", "quantity", "price"])
    df = load_data()
    #dickey_fuller_test(df, matnr=101728)
    import os
    dir = "/home/aman/PycharmProjects/seasonality_hypothesis/older_plots/plots_product_aggregate/"
    for i in os.listdir((dir)):
        try:
            dickey_fuller_test(df, matnr=int(i.split("_")[0]))
        except:
            pass