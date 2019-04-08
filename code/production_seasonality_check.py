import pandas as pd
import pickle

pickle_in = open("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/raw_data_agg_out_c005d.pickle", "rb")
raw_data = pickle.load(pickle_in)
kunag = 500056565
matnr = 100278
data = raw_data[raw_data["kunag_matnr"] == (kunag, matnr)]
print(data.columns)
print(data["time_series"])
