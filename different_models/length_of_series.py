import pandas as pd
import data_transformation

def duration_of_series(input_df, kunag, matnr):
    """
    returns number of days between start and end-date of the selected series
    param : dataframe and strings
    return : integer
    """
    input_df = input_df[(input_df["kunag"]==kunag) & (input_df["matnr"]==matnr)]
    input_df["parse_date"] = input_df["date"].apply(lambda date:pd.to_datetime(date, format = "%Y%m%d"))
    input_df = input_df.sort_values("date")
    start_date = input_df.iloc[0]["parse_date"]
    end_date = input_df.iloc[-1]["parse_date"]
    duration_in_days = (end_date - start_date).days
    return duration_in_days, input_df.shape[0] 

if __name__ == "__main__":

    import pandas as pd
    import data_transformation
    import framework
    from tqdm import tqdm

    df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv", sep = "\t")
    df = framework.remove_negative_rows(df)

    combination_with_freq = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/cleveland_norm_freq_last_year_all_series.csv")
    bucket_1 = combination_with_freq[combination_with_freq["norm_freq_last_year_all_series"]>26]

    num = 0
    total_mse = 0
    for i in [8,12,16,20,24,28,32,36,40]:
        for index, row in bucket_1.iterrows():
            kunag = int(row["kunag"])
            matnr = int(row["matnr"])
        
            if (duration_of_series(df, kunag, matnr )[0] >= 548):
                num+=1
                total_mse+=framework.moving_average(df, kunag, matnr,i)[1]
            if (num == 100):break

        print("lag:", i,", total_mse :", total_mse)