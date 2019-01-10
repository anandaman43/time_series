import pandas as pd
from datetime import datetime
import data_transformation

def possible_combinations(input_df):
    """
    """

    kunad_unique_values = input_df["kunag"].unique()
    unique_matnr_in_kunad = {}
    for i in kunad_unique_values:
        unique_matnr_in_kunad[i] = input_df[input_df["kunag"] == i]["matnr"].unique()
    return unique_matnr_in_kunad

def num_of_datapoints(input_df,name):
    """
    """

    datapoints = pd.DataFrame()
    unique_matnr_in_kunad = possible_combinations(input_df)
    for i in unique_matnr_in_kunad.keys():
        for k in unique_matnr_in_kunad[i]:
            ts = input_df[input_df["kunag"] == i][input_df[input_df["kunag"] == i]["matnr"]==k]["quantity"]
            datapoints = datapoints.append([[i,k,ts.shape[0]]],ignore_index=True)
    datapoints.columns = ["kunag","matnr","num_of_datapoints"]
    datapoints = datapoints.sort_values("num_of_datapoints",ascending=False)
    datapoints.to_csv("/home/aman/Desktop/CSO_drug/file_generated/" + name + ".csv",index=False)

def num_of_unique_datapoints(input_df,name):
    """
    """

    datapoints = pd.DataFrame()
    unique_matnr_in_kunad = possible_combinations(input_df)
    for i in unique_matnr_in_kunad.keys():
        for k in unique_matnr_in_kunad[i]:
            freq_2016 = datapoints_in_a_year(input_df, kunag = i, matnr = k, year = "2016")
            freq_2017 = datapoints_in_a_year(input_df, kunag = i, matnr = k, year = "2017")
            freq_2018 = datapoints_in_a_year(input_df, kunag = i, matnr = k, year = "2018")
            datapoints = datapoints.append([[i, k, freq_2016, freq_2017, freq_2018]],ignore_index=True)
    datapoints.columns = ["kunag","matnr","num_of_datapoints_in_2016", "num_of_datapoints_in_2017", "num_of_datapoints_in_2018"]
    datapoints = datapoints.sort_values("num_of_datapoints_in_2017", ascending=False)
    datapoints.to_csv("/home/aman/Desktop/CSO_drug/file_generated/" + name + ".csv",index=False)


def remove_negative(input_df):
    """ rows having negative values (quantity) are dropped """
    output_df = input_df[input_df["quantity"] >= 0]
    return output_df

def select_series(input_df, kunag = 600142082, matnr = 115583):
    """ selects a series corresponding to the given kunag and matnr """
    input_df = remove_negative(input_df)
    output_ts = input_df[input_df["kunag"] == kunag][input_df[input_df["kunag"] == kunag]["matnr"]==matnr]
    return output_ts

def datapoints_in_a_year(input_df, kunag = 600142082, matnr = 115583, year = "2017"):
    """ number of positive datapoints in a series in year 2017 """

    input_df = remove_negative(input_df)
    ts = select_series(input_df, kunag = kunag, matnr = matnr)
    try:
        num_points = len(ts[year].index.drop_duplicates())
    except:
        num_points = 0
    return num_points

def frequency(input_df):
    """ returns number of datapoints in last year """

    input_df = remove_negative(input_df)
    try:
        latest_date = input_df.index[-1]
    except:
        return 0
    latest_year = latest_date.year-1
    latest_month = latest_date.month
    latest_day = latest_date.day

    last_year_date = datetime(latest_year, latest_month, latest_day)
    output_df = input_df[last_year_date:]
    freq = output_df.shape[0]
    return freq

def freq_last_year_all_series(input_df, name="freq_last_year_all_series"):
    
    datapoints = pd.DataFrame()
    unique_matnr_in_kunad = possible_combinations(input_df)
    for i in unique_matnr_in_kunad.keys():
        for k in unique_matnr_in_kunad[i]:
            ts = select_series(input_df, kunag=i, matnr=k)
            datapoints = datapoints.append([[i,k,frequency(ts)]],ignore_index=True)
    datapoints.columns = ["kunag","matnr","freq_last_year_all_series"]
    datapoints = datapoints.sort_values("freq_last_year_all_series",ascending=False)
    datapoints.to_csv("/home/aman/Desktop/CSO_drug/file_generated/" + name + ".csv",index=False)

def norm_freq_last_year_all_series(input_df, name="norm_freq_last_year_all_series"):
    
    datapoints = pd.DataFrame()
    unique_matnr_in_kunad = possible_combinations(input_df)
    for i in unique_matnr_in_kunad.keys():
        for k in unique_matnr_in_kunad[i]:
            ts = select_series(input_df, kunag=i, matnr=k)
            datapoints = datapoints.append([[i,k,normalized_frequency(ts)]],ignore_index=True)
    datapoints.columns = ["kunag","matnr","norm_freq_last_year_all_series"]
    datapoints = datapoints.sort_values("norm_freq_last_year_all_series",ascending=False)
    datapoints.to_csv("/home/aman/Desktop/CSO_drug/file_generated/" + name + ".csv",index=False)
    return datapoints


def normalized_frequency(input_df):
    """ returns number of datapoints in last year with normalization 
        input: a ts with a particular kunag and matnr
        output: an integer    
    """
    input_df = remove_negative(input_df)
    try:
        latest_date = input_df.index[-1]
        first_date = input_df.index[0]
    except:
        return 0
    latest_year = latest_date.year-1
    latest_month = latest_date.month
    latest_day = latest_date.day

    last_year_date = datetime(latest_year, latest_month, latest_day)
    diffrnce_in_days = (first_date - last_year_date).days
    year_diffrnce_in_days  = (latest_date - last_year_date).days
    # print("year_diffrnce_in_days",year_diffrnce_in_days)
    if diffrnce_in_days<=0 :
        output_df = input_df[last_year_date:]
        freq = output_df.shape[0]
        return freq
    elif diffrnce_in_days>=(year_diffrnce_in_days-90):
        return 0   # putting freq = 0 for series having data only for last 3 months (90 days) or less
    else:
        output_df = input_df[last_year_date:]
        freq = output_df.shape[0]
        multiplier = float(year_diffrnce_in_days)/(float(year_diffrnce_in_days)-diffrnce_in_days)
        normalized_freq = float(freq)*multiplier
        return normalized_freq


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf

    file_address = "/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv"

    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    data = pd.read_csv(file_address, sep = "\t", parse_dates=['date'], index_col='date',date_parser=dateparse)
    data = data.sort_index()
    #norm_freq_last_year_all_series(data)
    ts = select_series(data, kunag=500076413, matnr=144089)
    ts = ts.reset_index()
    #ts.to_csv("original.csv")
    ts = data_transformation.get_weekly_aggregate(ts) 
    #ts.to_csv("aggregated.csv")
    #pd.DataFrame(acf(ts["quantity"])).plot(kind='bar') 
    #plt.show()  
    #ts.to_csv("/home/aman/Desktop/CSO_drug/file_generated/series.csv")
    print("done")

    # plt.plot(ts_1)
    # plt.show()

   
