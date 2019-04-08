__doc__ = """this code provides error for a dataframe cotaining predicted and actual values"""

import pandas as pd


def local_error(input_df, num_points=16, plot=False):
    """
    calculates local demand for last num_points of data and provides rmse
    :param input_df:
    :param num_points:
    :return:
    """
    input_df_copy = input_df.copy()
    input_df_copy = input_df_copy.iloc[-num_points:]
    actual = input_df_copy["actual"].values
    predicted = input_df_copy["predicted"].values

import numpy as np

a = [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
actual = np.array(a)
print([actual > 0].index
