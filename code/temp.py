from outlier import ma_replace_outlier
from selection import load_data
from selection import remove_negative_rows
from data_transformation import get_weekly_aggregate
from outlier import ma_replace_outlier
from dateutil import parser
df = load_data()
df = remove_negative_rows(df)
df = df[df["date"] >= 20160704]
df = get_weekly_aggregate(df)
_testing = df[["quantity", "dt_week"]].copy()
aggregated_data = _testing.rename(columns={'dt_week': 'ds', 'quantity': 'y'})
aggregated_data.ds = aggregated_data.ds.apply(str).apply(parser.parse)
aggregated_data.y = aggregated_data.y.apply(float)
aggregated_data = aggregated_data.sort_values('ds')
aggregated_data = aggregated_data.reset_index(drop=True)
_result = ma_replace_outlier(data=aggregated_data, n_pass=3, aggressive=True, window_size=12, sigma=3.0)
_result[0].to_csv("/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_outlier_removed.csv")
_result[0].groupby("ds")["y"].sum().to_csv(
    "/home/aman/PycharmProjects/seasonality_hypothesis/data_generated/groupby_aggregated_outlier_removed.csv")