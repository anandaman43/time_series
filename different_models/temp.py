# import framework
# import transformation
# import data_transformation
import pandas as pd
# from tqdm import tqdm
#
# df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv", sep = "\t")
#
# total_mse = 0
#
# combination_with_freq = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/cleveland_norm_freq_last_year_all_series.csv")
# bucket_1 = combination_with_freq[combination_with_freq["norm_freq_last_year_all_series"]>26]
#
# for index, row in tqdm(bucket_1.iterrows()):
#     kunag = int(row["kunag"])
#     matnr = int(row["matnr"])
#     try:
#         # print(total_mse)
#         total_mse+=framework.holts_winter_method(df, kunag, matnr)[1]
#         # print(total_mse)
#     except:
#         print(kunag, matnr)
#
# print(total_mse)


# df = framework.remove_negative_rows(df)
# df_series = framework.individual_series(df, 500057582   ,152222)
# a = df_series.copy()
# df_series = data_transformation.get_weekly_aggregate(df_series)
# b = df_series.copy()
# df_series["date"] = df_series["dt_week"].map(str)
# df_series["date"] = df_series["date"].apply(lambda x:x.replace("-",""))
# df_series["prediction"] = df_series["quantity"]
# framework.output_plot(df_series, 0.12)

# import itertools
# for i in itertools.product([1,2,3], [4,5,6], [7,8,9]):
#     print(i)

data = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/frequency_cleaveland.csv")
print(data["frequency"].value_counts())
