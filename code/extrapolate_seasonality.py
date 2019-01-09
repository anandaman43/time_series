import pandas as pd


def season_week_extension(input_df):
    season_df = input_df.copy()
    season_df["dt_week"] = season_df["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    last_week = season_df.iloc[season_df.shape[0]-1]["dt_week"]
    next_week =pd.to_timedelta(7, unit='D') + last_week
    next_week_quantity = season_df.iloc[season_df.shape[0] - 52]["quantity"]
    next_row = pd.DataFrame([[next_week, next_week_quantity]], columns=["dt_week", "quantity"])
    season_df = season_df.append(next_row, ignore_index=True)
    return season_df


def season_extrapolate(input_df, num_weeks=1):
    df_copy = input_df.copy()
    for i in range(num_weeks):
        df_copy = season_week_extension(df_copy)
    return df_copy


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    season = pd.read_csv(
        "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed_seas.csv",
    names=["dt_week", "quantity"])
    season = season_extrapolate(season, 52)
    season = season.set_index("dt_week")
    plt.plot(season["quantity"], marker=".", label='extrapolated_52_weeks')
    plt.legend(loc="upper left")
    plt.xlabel("dt_weeks")
    plt.ylabel("aggregated quantities")
    plt.title("2019_extrapolation")
    plt.show()
