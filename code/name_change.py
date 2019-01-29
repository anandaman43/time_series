import os
from seasonality_detection import ljung_box_test
from selection import load_data
df = load_data()
k=0
for i in os.listdir("/home/aman/PycharmProjects/seasonality_hypothesis/latest_plots"):
    matnr = int(i.split("_")[0])
    seas_pres = ljung_box_test(df, matnr)
    os.rename("/home/aman/PycharmProjects/seasonality_hypothesis/latest_plots/"+i,
              "/home/aman/PycharmProjects/seasonality_hypothesis/latest_plots/"+str(seas_pres)+"_"+i)
    k += 1
    print(k)