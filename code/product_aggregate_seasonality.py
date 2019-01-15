import pandas as pd
from tqdm import tqdm
from data_transformation import get_weekly_aggregate
from dateutil import parser
from outlier import ma_replace_outlier
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def overall_aggregate_seas(input_df, matnr=103029):
    """
    This function aggregates whole cleaveland data with ma outliers removing different categories series outliers
    First week has been removed
    :return: pandas_df : seasonal component of the aggregated data
    """
    df = input_df.copy()
    df = df[df["matnr"] == matnr]
    overall = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C025.csv")
    overall = overall[overall["matnr"] == matnr]
    product = pd.read_csv("~/PycharmProjects/seasonality_hypothesis/data/material_list.tsv", sep="\t")
    product_name = product[product["matnr"] == str(matnr)]["description"].values[0]
    k = 0
    for index, row in overall.iterrows():
        frequency = row["frequency"]
        days = row["days"]
        df_series = df[(df["kunag"] == row["kunag"]) & (df["matnr"] == row["matnr"])]
        df_series = df_series[df_series["quantity"] >= 0]
        df_series = df_series[df_series["date"] >= 20160703]
        if int(frequency) == 0:
            continue
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

    try:
        final = final.groupby("dt_week")["quantity"].sum().reset_index()
    except:
        return None
    final = final.set_index("dt_week")
    print(final["quantity"])
    print(final["quantity"].shape)
    result = seasonal_decompose(final["quantity"], model="additive")
    result.plot()
    #plt.show()
    plt.savefig(
        "/home/aman/PycharmProjects/seasonality_hypothesis/plots_product_C0025/"+str(matnr)+"_"+product_name+".png")
    #result.seasonal.to_csv(
    #    "~/PycharmProjects/seasonality_hypothesis/data_generated/product_aggregate_seasonality_"+str(matnr)+".csv")
    return result.seasonal


if __name__ == "__main__":
    df = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data/4200_C025_raw_invoices_2019-01-06.tsv",
                     names=["kunag", "matnr", "date", "quantity", "price"])

    #"""
    frequency_days = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_days_4200_C025.csv")
    products = frequency_days["matnr"].unique()
    error = []
    for product in tqdm(products):
        try:overall_aggregate_seas(df, product)
        except:
            error.append(product)
            print(product)
    for k in error:
        print(k)
        
    #"""

    #overall_aggregate_seas(df, 102580)

