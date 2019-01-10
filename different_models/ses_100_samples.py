import framework
import transformation
import data_transformation
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("/home/aman/Desktop/CSO_drug/data/raw_invoices_cleaveland_sample_100_stores_2018-12-09.tsv", sep = "\t")

total_mse = 0

combination_with_freq = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/cleveland_norm_freq_last_year_all_series.csv")
bucket_1 = combination_with_freq[combination_with_freq["norm_freq_last_year_all_series"]>26]

sample_results = pd.DataFrame(index = range(0,5))

values = {}
for k in tqdm([0.1, 0.2, 0.3, 0.4, 0.5]):
    sample_results[str(k)] = 0
    for random_state in [1,2,3,4,5]:
        total_mse = 0
        for index, row in bucket_1.sample(100, random_state=random_state).iterrows():
            kunag = int(row["kunag"])
            matnr = int(row["matnr"])
            try:
                # print(total_mse)
                total_mse+=framework.simple_exponential_smoothing(df, kunag, matnr,k)[1]
                # print(total_mse)
            except:
                
                ts = framework.individual_series(df, kunag, matnr).sort_values("date")
                first_time = ts.loc[ts.index[0]]["date"]
                last_time = ts.loc[ts.index[-1]]["date"]
                print(kunag, matnr, framework.individual_series(df, kunag, matnr).shape[0], first_time, last_time)
        sample_results[str(k)].iloc[random_state-1] = total_mse
        # print(k, total_mse)

print(sample_results.mean())
        