from selection import load_data
from selection import individual_series
import pandas as pd
import matplotlib.pyplot as plt


df = load_data()
series = individual_series(df).set_index("dt_week")
series1 = series["2016-07-07": "2016-09-15"]
series2 = series1.copy()
series3 = series1.copy()
series2["quantity"]["2016-08-25": "2016-09-15"] = 4
series3["quantity"]["2016-08-25": "2016-09-15"] = 5
plt.figure(figsize=(16,8))
plt.plot(series1["quantity"], marker="o", label='series1')
plt.plot(series2["quantity"], marker="o", label='series2')
plt.plot(series3["quantity"], marker="o", label='series3')
plt.show()