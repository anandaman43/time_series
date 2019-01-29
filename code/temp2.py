from selection import individual_series
from preprocess import splitter_2
import pandas as pd
from selection import load_data
from outlier import ma_replace_outlier
from dateutil import parser
df_series = individual_series(df)
train = splitter_2(df_series)[0]
_testing = train[["quantity", "dt_week"]].copy()
aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})

aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
aggregated_data.y = aggregated_data.y.apply(float)
aggregated_data = aggregated_data.sort_values('ds')
aggregated_data = aggregated_data.reset_index(drop=True)
print(aggregated_data)
_result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
print(_result)