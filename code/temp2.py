# from selection import individual_series
# from preprocess import splitter_2
# import pandas as pd
# from selection import load_data
# from outlier import ma_replace_outlier
# from dateutil import parser
# df_series = individual_series(df)
# train = splitter_2(df_series)[0]
# _testing = train[["quantity", "dt_week"]].copy()
# aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})
#
# aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
# aggregated_data.y = aggregated_data.y.apply(float)
# aggregated_data = aggregated_data.sort_values('ds')
# aggregated_data = aggregated_data.reset_index(drop=True)
# print(aggregated_data)
# _result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
# print(_result)


# ----------------------------------for seeing wrong prediction--------------------------------------------------- #

# import pandas as pd
# from shutil import copyfile
# old_path_main = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/"
# data = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/report_2018_02_12.csv")
# data = data[(data["dtw_flag"] == True) & (data["diff"] < 0)]
# for index, row in data.iterrows():
#     matnr = row["matnr"]
#     kunag = row["kunag"]
#     src = old_path_main + str(kunag) + "_" + str(matnr) + "_True.png"
#     dst = old_path_main + "wrong_pred/" + str(kunag) + "_" + str(matnr) + "_True.png"
#     copyfile(src, dst)

# -----------------------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
from shutil import copyfile

data11 = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_11/report_2018_02_11.csv")
data12 = pd.read_csv("/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/report_2018_02_12.csv")
c = pd.merge(data11, data12, on=["kunag", "matnr"])
c["change"] = np.sign(c["diff_x"]) == np.sign(c["diff_y"])
c_part_gone_wrong = c[(c["dtw_flag_x"] == True) & (c["diff_x"] >0) &(c["diff_y"]<0)]
for index, row in c_part_gone_wrong.iterrows():
    old_11 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_11/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True.png"
    old_12 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True.png"
    new_11 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/95_12_gone_wrong/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True_011.png"
    new_12 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/95_12_gone_wrong/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True_022.png"
    copyfile(old_11, new_11)
    copyfile(old_12, new_12)
print(c_part_gone_wrong)
c_part_gone_correct = c[(c["dtw_flag_x"] == True) & (c["diff_x"] <0) &(c["diff_y"]>0)]
for index, row in c_part_gone_correct.iterrows():
    old_11 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_11/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True.png"
    old_12 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True.png"
    new_11 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/57_17_gone_correct/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True_011.png"
    new_12 = "/home/aman/PycharmProjects/seasonality_hypothesis/report_2018_02_12/57_17_gone_correct/" + str(row["kunag"]) + "_" +\
             str(row["matnr"]) + "_True_022.png"
    copyfile(old_11, new_11)
    copyfile(old_12, new_12)
print(c_part_gone_correct)