import pandas as pd
from datetime import datetime
from transformation import remove_negative
import transformation


def frequency(input_df):
    """
    returns number of datapoints in last year
    :param input_df:
    :return:
    """

    input_df = transformation.remove_negative(input_df)
    latest_date = input_df.index[-1]
    latest_year = latest_date.year-1
    latest_month = latest_date.month
    latest_day = latest_date.day

    last_year_date = datetime(latest_year, latest_month, latest_day)
    output_df = input_df[last_year_date:]
    freq = output_df.shape[0]
    return freq



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import transformation
    import data_transformation

    file_address = "/home/aman/Desktop/CSO_drug/data/raw_data_drug_store_sample_30_2018-12-10.tsv"

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    data = pd.read_csv(file_address, sep = "\t", parse_dates=['date'], index_col='date',date_parser=dateparse)
    data = data.sort_index()
    data = transformation.remove_negative(data)
    ts = transformation.select_series(data, kunag=600142082, matnr=145105)
    ts = ts.reset_index()
    ts = data_transformation.get_weekly_aggregate(ts)
    ts.Timestamp = pd.to_datetime(ts.dt_week,format='%Y-%m-%d') 
    ts.index = ts.Timestamp
    plt.figure(figsize=(12,8))
    plt.plot(ts["quantity"])
    plt.show()
    plt.savefig("abc.png")
    # print(ts)
    # print("done")
    norm_freq_last_year_all_series(data)
    combination_with_freq = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/freq_last_year_all_series.csv")
    bucket_1 = combination_with_freq[combination_with_freq["freq_last_year_all_series"]>26]
    bucket_2 = combination_with_freq[(combination_with_freq["freq_last_year_all_series"]>20) & (combination_with_freq["freq_last_year_all_series"]<=26)]
    bucket_3 = combination_with_freq[(combination_with_freq["freq_last_year_all_series"]>12) & (combination_with_freq["freq_last_year_all_series"]<=20)]
    bucket_4 = combination_with_freq[combination_with_freq["freq_last_year_all_series"]<=12]
    print(bucket_1.shape[0])
    print(bucket_2.shape[0])
    print(bucket_3.shape[0])
    print(bucket_4.shape[0])   