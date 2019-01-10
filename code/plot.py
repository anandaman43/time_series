import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def plot(df):
    df["dt_week"] = df["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    df = df.set_index("dt_week")
    plt.plot(df["quantity"], marker=".", label='overall data aggregated')
    plt.legend(loc="upper left")
    plt.xlabel("dt_weeks")
    plt.ylabel("aggregated quantities")
    plt.title("aggregated")
    plt.show()s


if __name__ == "__main__":
        season = pd.read_csv(
                "~/PycharmProjects/seasonality_hypothesis/data_generated/aggregated_complete_outliers_removed.csv")
        season["dt_week"] = season["dt_week"].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
        season = season.set_index("dt_week")
        result = seasonal_decompose(season["quantity"], model="additive")
        result.plot()
        plt.show()
        # plt.plot(season["quantity"], marker=".", label='overall data aggregated')
        # plt.legend(loc="upper left")
        # plt.xlabel("dt_weeks")
        # plt.ylabel("aggregated quantities")
        # plt.title("aggregated")
        # plt.show()