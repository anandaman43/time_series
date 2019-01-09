import pandas as pd


def smoothing(df):
    df_copy = df.copy()
    max_index = df.shape[0] - 1
    for i in range(max_index-1):
        mean = df.iloc[i:i+3]["quantity"].mean()
        df_copy["quantity"].iloc[i+1] = mean
    df_copy["quantity"].iloc[0] = df.iloc[0:2]["quantity"].mean()
    df_copy["quantity"].iloc[-1] = df.iloc[-2:]["quantity"].mean()
    return df_copy


def smoothing_5(df):
    df_copy = df.copy()
    max_index = df.shape[0] - 1
    for i in range(0, max_index-3):
        mean = df.iloc[i:i+5]["quantity"].mean()
        df_copy["quantity"].iloc[i+2] = mean
    df_copy["quantity"].iloc[0] = df.iloc[0:2]["quantity"].mean()
    df_copy["quantity"].iloc[1] = df.iloc[0:3]["quantity"].mean()
    df_copy["quantity"].iloc[-1] = df.iloc[-2:]["quantity"].mean()
    df_copy["quantity"].iloc[-2] = df.iloc[-3:]["quantity"].mean()
    return df_copy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed_seas.csv",
        names=["dt_week", "quantity"])

    df_1 = smoothing(data)
    df_2 = smoothing_5(data)
    data["dt_week"] = data["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    data = data.set_index("dt_week")
    df_1["dt_week"] = df_1["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    df_1 = df_1.set_index("dt_week")
    df_2["dt_week"] = df_2["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    df_2 = df_2.set_index("dt_week")
    #plt.plot(df_1["quantity"], marker=".", label=' 3 point average')
    plt.plot(df_2["quantity"], marker=".", label=' 5 point average')
    #plt.plot(data["quantity"], marker=".", label='overall data aggregated')
    plt.legend(loc="upper left")
    plt.xlabel("dt_weeks")
    plt.ylabel("aggregated quantities")
    plt.title("5 point average")
    plt.show()
