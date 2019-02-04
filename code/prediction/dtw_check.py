from dtw import dtw
from selection import load_data
from selection import individual_series
from preprocess import splitter_2
from stl_decompose import product_seasonal_comp_7_point
from smoothing import smoothing_5_new
import matplotlib.pyplot as plt
import pandas as pd
import os


def dtw_check(df, kunag, matnr, threshold=0.18):
    df_series = individual_series(df, kunag, matnr)
    df_series = smoothing_5(df_series)
    df_series = df_series.set_index("dt_week")
    series_norm = (df_series - df_series.mean()) / df_series.std()
    seasonality_product = product_seasonal_comp_7_point(df, matnr)
    seasonality_req_subset = seasonality_product.loc[df_series.index]
    seasonality_req_subset_norm = (seasonality_req_subset - seasonality_req_subset.mean())/seasonality_req_subset.std()
    l2_norm = lambda x, y: (x - y) ** 2
    x = series_norm["quantity"]
    y = seasonality_req_subset_norm["quantity"]
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=l2_norm)
    if d <= threshold:
        return True
    else:
        return False


if __name__=="__main__":

    df = load_data()
    print(dtw_check(df, 500083147, 135573))
    # path = "/home/aman/PycharmProjects/seasonality_hypothesis/stl_plots_seasonality_108_7_point_thresh_0.01"
    # result = pd.DataFrame()
    # count = 0
    # for i in os.listdir(path):
    #     kunag = int(i.split("_")[0])
    #     matnr = int((i.split("_")[1]).split(".")[0])
    #     df_series = individual_series(df, kunag, matnr)
    #     plot1 = df_series.set_index("dt_week")["quantity"]
    #     plt.figure(figsize=(16, 5))
    #     plt.plot(plot1, marker=".", label="original")
    #     df_series = smoothing_5(df_series)
    #     plot2 = df_series.set_index("dt_week")
    #     plot2_norm = (plot2-plot2.mean())/plot2.std()
    #     plt.plot(plot2["quantity"], marker=".", label="smoothened")
    #     plt.plot(plot2_norm["quantity"], marker=".", label="smoothened normalized")
    #     seasonality_product = product_seasonal_comp_7_point(df, matnr)
    #     #print(df_series)
    #     #print(seasonality_product)
    #     seasonality = seasonality_product.loc[df_series.set_index("dt_week").index]
    #     plot3 = seasonality
    #     plot3_norm = (plot3-plot3.mean())/plot3.std()
    #     plt.plot(plot3_norm["quantity"], marker=".", label="seasonality")
    #     #print(seasonality)
    #     l2_norm = lambda x, y: (x - y) ** 2
    #     x = plot2_norm["quantity"]
    #     y = plot3_norm["quantity"]
    #     d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=l2_norm)
    #     result = result.append([[kunag, matnr, d]])
    #     plt.legend()
    #     plt.title(str(d))
    #     plt.savefig("/home/aman/PycharmProjects/seasonality_hypothesis/dtw/"+str(matnr)+"_"+str(kunag)+".png")
    #     count += 1
    #     print(count)
    #     # print(result)
    # result.columns = ["kunag", "matnr", "d"]
    # result.to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/stl_plots_seasonality_108_7_point_thresh_0.01/result.csv")

