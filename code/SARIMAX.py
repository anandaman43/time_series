import pandas as pd
import requests
from io import BytesIO
import statsmodels.api as sm
friedman2 = requests.get('https://www.stata-press.com/data/r12/friedman2.dta').content
raw = pd.read_stata(BytesIO(friedman2))
raw.index = raw.time
data = raw.loc[:'1981']
print(data)
endog = data.loc['1959':, 'm2']
exog = sm.add_constant(data.loc['1959':, 'm2'])
nobs = endog.shape[0]
print(endog)
print(exog)