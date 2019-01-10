import pandas as pd
from selection import load_data
from selection import select_series
from data_transformation import get_weekly_aggregate





if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    from models import remove_negative_rows
    from models import individual_series
    from models import splitter
    import matplotlib.pyplot as plt
    f
    df = load_data()
    # df = remove_negative_rows(df)
    # ts = individual_series(df, kunag=500057582, matnr=103029)
    # print(ts)
    # ts = get_weekly_aggregate(ts)
    # ts["date"] = ts["dt_week"].map(str)
    # ts["date"] = ts["date"].apply(lambda x: x.replace("-", ""))
    # ts["prediction"] = ts["quantity"]
    # ts1, ts2 = splitter(ts)
    # ts = prophet_preprocess(ts1)
    # m = Prophet()
    # m.fit(ts1)
    # future = m.make_future_dataframe(periods=1, freq="W", include_history=False).apply(
    #     lambda x: (x + pd.Timedelta(4, unit="D")))  # + pd.Timedelta(4, unit="D")
    # print(ts1)
    # forecast = m.predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    # print(ts2)
    # fig1 = m.plot(forecast)
    # plt.show()



