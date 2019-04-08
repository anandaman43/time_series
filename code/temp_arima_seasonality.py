import statsmodels.api as sm
import pandas as pd
import numpy as np
from MannKendallTrend.mk_test import mk_test

d = {"dt_week": [pd.datetime(2018, 12, 1), pd.datetime(2018, 12, 8), pd.datetime(2018, 12, 15),
                 pd.datetime(2018, 12, 22), pd.datetime(2018, 12, 29), pd.datetime(2019, 1, 5),
                 pd.datetime(2019, 1, 12), pd.datetime(2019, 1, 19), pd.datetime(2019, 1, 26),
                 pd.datetime(2019, 2, 2), pd.datetime(2019, 1, 19)],
     "quantity": [1, 2, 0, 6, 6, 6, np.nan, 8, 6, 2, 5]
}


df = pd.DataFrame(d)
print(df["quantity"].isnull().sum())

# def func(x1, x2):
#     # print(x1, x2)
#     return  x2
# print(df.apply(lambda row: func(row["dt_week"], row["quantity"]), axis=1))
# print(df)
# print(df)
# period = 4
# material_data_detrended_smooth = df
# length = material_data_detrended_smooth.shape[0]
# extra = int(length % period)
# material_data_detrended_smooth = material_data_detrended_smooth.iloc[extra:length]
# print(material_data_detrended_smooth)
# length = material_data_detrended_smooth.shape[0]
# material_data_detrended_smooth = material_data_detrended_smooth.reset_index(drop=True)
# material_data_agg = material_data_detrended_smooth.copy()
# for i in range(0, length, period):
#     material_data_agg["quantity"].iloc[i] = material_data_detrended_smooth.iloc[i:i+period]["quantity"].sum()
#     material_data_agg = material_data_agg.drop(list(range(i + 1, i + period)), axis=0)
# material_data_agg = material_data_agg.set_index("dt_week")
# print(material_data_agg)
"""
df = df.set_index("dt_week")

train = df.iloc[0:7]
print(train)
pdq = (0, 1, 1)
seasonal_pdq = (0, 0, 0, 52)
mod = sm.tsa.statespace.SARIMAX(train, order=pdq, seasonal_order=seasonal_pdq,
                                            enforce_stationarity=False, enforce_invertibility=False,
                                            measurement_error=False, time_varying_regression=False,
                                            mle_regression=True)

results = mod.fit(disp=False)
##########################################################################

##########################################################################
# generating forecast for test data points
##########################################################################
print("start: ", pd.to_datetime(np.amax(train.index)))
print("end: ", pd.to_datetime(np.amax(df.index)))
pred_test = results.get_prediction(start=pd.to_datetime(np.amax(train.index)),
                                   end=pd.to_datetime(np.amax(df.index)), dynamic=True)
print(pred_test.predicted_mean)

"""