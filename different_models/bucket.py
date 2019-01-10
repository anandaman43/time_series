from selection import load_data
from frequency import normalized_frequency


def bucket():
    df = load_data()
    frequency_cleaveland = pd.DataFrame()
    for index, group in df.groupby(["kunag", "matnr"]):
        ts = select_series(df, index[0], index[1])
        freq = normalized_frequency(ts)
        frequency_cleaveland = frequency_cleaveland.append([[index[0], index[1], freq]])
    frequency_cleaveland.columns = ["kunag", "matnr", "frequency"]
    frequency_cleaveland.to_csv("/home/aman/Desktop/CSO_drug/file_generated/frequency_cleaveland.csv")
    frequency_cleaveland = pd.read_csv("/home/aman/Desktop/CSO_drug/file_generated/frequency_cleaveland.csv")
