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


def smoothing_before_aggregation(input_df):
    df_copy = input_df.copy()
    max_index = input_df.shape[0] - 1
    for i in range(0, max_index-3):
        mean = input_df.iloc[i:i+5]["quantity"].mean()
        df_copy["quantity"].iloc[i+2] = mean
    return df_copy.iloc[2:-2]


def monthly_aggregate(detrended_df, period=4):
    # print("input :\n", detrended_df)
    input_df = detrended_df.copy()
    input_df = smoothing_before_aggregation(input_df)
    length = input_df.shape[0]
    input_df = input_df.reset_index()
    output_df = input_df.copy()
    for i in range(0, length, period):
        output_df["quantity"].loc[i] = input_df.iloc[i:i+period]["quantity"].sum()
        output_df = output_df.drop(list(range(i + 1, i + period)), axis=0)
    output_df = output_df.set_index("dt_week")
    # plt.plot(input_df.set_index("dt_week"), marker=".", label="before monthly aggregatation")
    # plt.plot(output_df, marker=".", label="monthly aggregate")
    # plt.legend()
    # plt.show()
    return output_df


def detrend(input_df, order=24):
    input_df_copy = input_df.copy()
    input_df_copy = input_df_copy.reset_index()
    length = input_df_copy.shape[0]
    output_df = input_df_copy.copy()
    for i in range(order, length-order):
        output_df["quantity"].iloc[i] = input_df_copy.iloc[i-order:i+order+1]['quantity'].mean()
    output_df = output_df.iloc[order:length-order]
    output_df = output_df.set_index("dt_week")
    input_df_copy = input_df_copy.iloc[order:length-order].set_index("dt_week")
    detrended = input_df_copy - output_df
    # print(input_df_copy)
    # print(output_df)
    # print(input_df_copy - output_df)
    # plt.plot(input_df_copy, marker=".", label="before removing trend")
    # plt.figure(figsize=(16, 8))
    # plt.plot(output_df, marker=".", markerfacecolor="red", label="y")
    # plt.plot(detrended, marker=".", label="after removing trend")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel("Date", fontsize=14)
    # plt.ylabel("Quantity", fontsize=14)
    # plt.title("Trend", fontsize=16)
    # plt.legend(fontsize=14)
    # plt.show()
    return detrended


def outlier_on_aggregated(aggregated_df):
    _testing = aggregated_df[["quantity", "dt_week"]].copy()
    aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

    aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
    aggregated_data.y = aggregated_data.y.apply(float)
    aggregated_data = aggregated_data.sort_values('ds')
    aggregated_data = aggregated_data.reset_index(drop=True)
    n_pass = 3
    window_size = 12
    sigma = 3.0
    _result = ma_replace_outlier(data=aggregated_data, n_pass=n_pass, aggressive=True, window_size=window_size,
                                 sigma=sigma)
    result = _result[0].rename(columns={'ds': 'dt_week', 'y': 'quantity'})
    # plt.plot(result.set_index("dt_week")["quantity"], marker=".", label="after outlier")
    # # plt.plot(result.set_index("dt_week").diff(), marker=".", label="differenced after outlier")
    # plt.title("aggregated outlier removed")
    # plt.show()
    return result


def ljung_box_test(input_df, matnr=112260):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
    overall = overall[overall["matnr"] == matnr]
    # product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    # product_name = product[product["matnr"] == str(int(matnr))]["description"].values[0]
    k = 0
    for index, row in overall.iterrows():
        frequency = row["frequency"]
        days = row["days"]
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        if frequency == 0:
            continue
        # print(df_series)
        df_series = get_weekly_aggregate(df_series)
        # plt.plot(df_series.set_index("dt_week")["quantity"], marker=".", label="individual series")
        # plt.title(str(row["matnr"]) + " before outlier")
        # plt.show()
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
        # plt.plot(result.set_index("dt_week")["quantity"], marker=".", label="individual_series_after_outlier")
        # plt.title(str(row["matnr"]) + " after outlier")
        # plt.show()
        if k == 1:
            final = pd.concat([final, result])
        if k == 0:
            final = result
            k = 1
    final = final.groupby("dt_week")["quantity"].sum().reset_index()
    # plt.figure(figsize=(16, 8))
    # plt.plot(final.set_index("dt_week"), marker=".", markerfacecolor="red", label="y")
    # plt.plot(final.set_index("dt_week").diff(), marker=".", label="differenced aggregated_data")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel("Date", fontsize=14)
    # plt.ylabel("Quantity", fontsize=14)
    # plt.title("Product Weekly Aggregated Data", fontsize=16)
    # plt.legend(fontsize=14)
    # plt.show()
    final = outlier_on_aggregated(final)
    final_temp = final
    plt.figure(figsize=(16, 8))
    plt.plot(final.set_index("dt_week"), marker=".", markerfacecolor="red", label="y")
    # plt.plot(final.set_index("dt_week").diff(), marker=".", label="differenced aggregated_data")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel("Date", fontsize=14)
    # plt.ylabel("Quantity", fontsize=14)
    # plt.title("Product Weekly Aggregated Data Outlier Removed", fontsize=16)
    # plt.legend(fontsize=14)
    plt.show()
    # print("________", final.dtypes)
    final = final.set_index("dt_week")
    missing_more_24 = missing_data_detection(final)
    if missing_more_24:
        # print("data is missing for more than 6 months")
        return False, 1, 0, final, final_temp
    #temp = final
    # final = final.diff()
    # print("checking the length of aggregated series ...")
    final, Flag = cond_check(final)
    if Flag:
        # print("detrending the aggregated series ...")
        final_detrended = detrend(final)
        # plt.figure(figsize=(16, 8))
        # plt.plot(final_detrended, marker=".", markerfacecolor="red", label="y")
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.xlabel("Date", fontsize=14)
        # plt.ylabel("Quantity", fontsize=14)
        # plt.title("Detrended", fontsize=16)
        # plt.legend(fontsize=14)
        # plt.show()
        # print("monthly aggregating the aggregated series ...")
        final_aggregate = monthly_aggregate(final_detrended)
        # plt.figure(figsize=(16, 8))
        # plt.plot(final_aggregate, marker=".", markerfacecolor="red", label="y")
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.xlabel("Date", fontsize=14)
        # plt.ylabel("Quantity", fontsize=14)
        # plt.title("Monthly Aggregated", fontsize=16)
        # plt.legend(fontsize=14)
        # plt.show()
        # print("standard deviation is", final.std()/ final.mean())
        # print("performing ljung box test ...")
        result = acorr_ljungbox(final_aggregate["quantity"], lags=[13])
        # print(result)
        result_dickey = adfuller(final_aggregate["quantity"])
        # print("statistic: %f" %result[0])
        # print("p-value: %f" %result[1])
        # print("p_value is :", result[1][0])
        if result[1] < 0.02:
            # print(str(matnr)+" is seasonal")
            return True, result[1][0], result_dickey[1], final, final_temp
        else:
            # print(str(matnr) + " is not seasonal")
            return False, result[1][0], result_dickey[1], final, final_temp
    else:
        print("length of series is less than 112")
        return [False, "length is small", 0, final, final_temp]


def ljung_box_test_without_aggregation(input_df, matnr=112260):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C005.csv")
    overall = overall[overall["matnr"] == matnr]
    #product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    #product_name = product[product["matnr"] == str(int(matnr))]["description"].values[0]
    k = 0
    for index, row in overall.iterrows():
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
    result = acorr_ljungbox(final["quantity"], lags=[52])

        # print("statistic: %f" %result[0])
        # print("p-value: %f" %result[1])
    if result[1] < 0.01:
        #print(result[1])
        return True, result[1], final
    else:
        return False, result[1], final


def missing_data_detection(final):
    value_counts = final["quantity"].value_counts()
    value_more_than_24 = value_counts[value_counts >= 24].index
    final = final.reset_index()
    flag = False
    for value in value_more_than_24:
        final = final[final["quantity"] != value]
        final_2 = final["dt_week"].diff()
        count = (final_2 >= pd.Timedelta(168, unit="D")).sum()
        if count >= 1:
            flag = True
            print("data is missing")
    return flag


if __name__ == "__main__":
    from selection import load_data
    # df = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data/4200_C005_raw_invoices_2019-01-06.tsv",
    #                  names=["kunag", "matnr", "date", "quantity", "price"])
    df = load_data()
    from MannKendallTrend.mk_test import mk_test
    print(mk_test(ljung_box_test(df, matnr=103996)[-1]["quantity"]))
    # print(ljung_box_test(df, matnr=126583))
    # import os
    # dir = "/home/aman/PycharmProjects/seasonality_hypothesis/older_plots/plots_product_aggregate/"
    # for i in os.listdir((dir)):
    #     try:
    #         dickey_fuller_test(df, matnr=int(i.split("_")[0]))
    #     except:
    #         pass

    # data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/new_seasonality_code_check/result.csv")
    # for index, row in data.iterrows():
    #     print(row["matnr"])
    #     ljung_box_test(df, matnr=row["matnr"])