import statsmodels.api as sm
import numpy as np

file_address = "/home/aman/Desktop/CSO_drug/data/raw_data_drug_store_sample_30_2018-12-10.tsv"

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
data = pd.read_csv(file_address, sep = "\t", parse_dates=['date'], index_col='date',date_parser=dateparse)
data = data.sort_index()
data = data.reset_index()
data = data.set_index("date")
data.head()

ts = data[data["kunag"] == 600142082][data[data["kunag"] == 600142082]["matnr"]==115583][["quantity"]]
ts = ts.resample('MS').mean()
# plt.plot(ts)
# plt.show()
# pd.DataFrame(acf(ts)).plot(kind='bar')
# plt.show()

train = ts["2016":"2018"]
#test = ts["2018"]

sm.tsa.seasonal_decompose(train).plot()
result = sm.tsa.stattools.adfuller(train)
plt.show()