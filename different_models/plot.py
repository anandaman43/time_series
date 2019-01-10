import pandas as pd
import transformation


def plot_save(data, name, kunag=600142082, matnr=145105):
    ts = transformation.select_series(data, kunag=kunag, matnr=matnr)
    ts = ts.reset_index()
    ts = data_transformation.get_weekly_aggregate(ts)
    ts.Timestamp = pd.to_datetime(ts.dt_week,format='%Y-%m-%d') 
    ts.index = ts.Timestamp
    plt.figure(figsize=(12,8))
    plt.plot(ts["quantity"])
    plt.savefig("/home/aman/Desktop/CSO_drug/plots/" + name + "_" + str(kunag) + "_" + str(matnr) + ".png")


if __name__=="__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import transformation
    import data_transformation

    #file_address = "/home/aman/Desktop/CSO_drug/data/raw_data_drug_store_sample_30_2018-12-10.tsv"
    file_address = "/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv"
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    data = pd.read_csv(file_address, sep = "\t", parse_dates=['date'], index_col='date',date_parser=dateparse)
    data = data.sort_index()
    combination_with_freq = transformation.norm_freq_last_year_all_series(data, name="cleveland_norm_freq_last_year_all_series")
    data = transformation.remove_negative(data)

    #combination_with_freq = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/norm_freq_last_year_all_series.csv")
    bucket_1 = combination_with_freq[combination_with_freq["norm_freq_last_year_all_series"]>26]
    bucket_2 = combination_with_freq[(combination_with_freq["norm_freq_last_year_all_series"]>20) & (combination_with_freq["norm_freq_last_year_all_series"]<=26)]
    bucket_3 = combination_with_freq[(combination_with_freq["norm_freq_last_year_all_series"]>12) & (combination_with_freq["norm_freq_last_year_all_series"]<=20)]
    bucket_4 = combination_with_freq[(combination_with_freq["norm_freq_last_year_all_series"]<=12) & (combination_with_freq["norm_freq_last_year_all_series"]>5)]
    print(bucket_1.shape[0])
    print(bucket_2.shape[0])
    print(bucket_3.shape[0])
    print(bucket_4.shape[0]) 

    sample_1 = bucket_1.sample(50, random_state=1)
    sample_2 = bucket_2.sample(50, random_state=1)
    sample_3 = bucket_3.sample(50, random_state=1)
    sample_4 = bucket_4.sample(50, random_state=1)

    for index,row in sample_1.iterrows():
        plot_save(data, "bucket_1", kunag=row["kunag"], matnr=row["matnr"])
    for index,row in sample_2.iterrows():
        plot_save(data, "bucket_2", kunag=row["kunag"], matnr=row["matnr"])
    for index,row in sample_3.iterrows():
        plot_save(data, "bucket_3", kunag=row["kunag"], matnr=row["matnr"])
    for index,row in sample_4.iterrows():
        plot_save(data, "bucket_4", kunag=row["kunag"], matnr=row["matnr"])
