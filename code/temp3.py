import pandas as pd
frequency_cleaveland = pd.read_csv(
        "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/frequency_cleaveland.csv")
bucket_greater_26 = frequency_cleaveland[frequency_cleaveland["frequency"] > 26]
print(bucket_greater_26.drop_duplicates(subset=["matnr"]).shape)
