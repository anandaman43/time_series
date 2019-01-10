from selection import load_data
from data_transform import get_weekly_aggregate
import matplotlib.pyplot as plt
import pandas as pd
df = load_data()[["date", "quantity"]]
df = df[(df["quantity"] >= 0) & (df["quantity"] <= 10)]
print(df["quantity"].value_counts()/df.shape[0])
aggregate_data = df.groupby("date")["quantity"].sum()
aggregate_data = aggregate_data.reset_index()
aggregate_data["date"] = aggregate_data["date"].apply(lambda x: pd.to_datetime(x, format="%Y%m%d"))
aggregate_data = aggregate_data.sort_values("date")
aggregate_data = aggregate_data.set_index("date")
plt.figure(figsize=(16,8))
plt.plot(aggregate_data["quantity"], label='quantity')
plt.title("aggregated plot")
plt.show()
