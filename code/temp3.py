import pandas as pd
from selection import load_data

df = load_data()
df = df [df["quantity"] >= 0]
matnr = 101728
df = df[df["matnr"] == matnr]
date1 = 20160731
date2 = 20160806
one_week = df[(df["date"]>= date1) & (df["date"] <= date2)]
# print(one_week)
print(df[df["kunag"] == 500283364])
