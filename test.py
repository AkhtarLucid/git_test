import pandas as pd
impotr numpy as np
import scipy.stats as stats
import pandas_profiling
from sklearn.metrics import mean_squared_error
%matplotlib inline


train =  pd.read_csv('/input/train.csv')
test = pd.read_csv('/input/test.csv')

print(train.head())
model = LinearRegression(train).fit()
model.predict(test)



print("Modelling is done successfully")

print('Happy Coding')



