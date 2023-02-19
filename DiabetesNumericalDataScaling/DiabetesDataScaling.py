### Scaling data
# This excersice is to compare and validate the effect of transforming data before apllying a model 


import timeit # measuring model's performance (execution time with transformed data or original data)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets, linear_model


X, y = datasets.load_diabetes(return_X_y=True)
raw = X[:, None, 2]  #Adjusting dimentions for preprocessing 


## Data Scaling Rules 
max_raw = max(raw)
min_raw = min(raw)
scaled = (2*raw - max_raw - min_raw)/(max_raw - min_raw)

fig, axs = plt.subplots(2, 1, sharex= True)

axs[0].hist(raw)
axs[1].hist(scaled)
plt.show()


## Training models 
def train_raw():
    linear_model.LinearRegression().fit(raw, y)

def trained_scaled():
    linear_model.LinearRegression().fit(scaled, y)


##Comparing processing times 
raw_time = timeit.timeit(train_raw, number= 100)
scaled_time = timeit.timeit(trained_scaled, number= 100)

print('train raw: {}'.format(raw_time))
print('train scaled: {}'.format(scaled_time))



##Non linear transformation 
df = pd.read_csv('cars.csv')
df.price_usd.hist()
plt.show()

#Transformatin with hyperbolic tan, tanh(x)
df.price_usd.apply(lambda x: np.tanh(x)).hist()
plt.show()

p= 10000 #calibrating
df.price_usd.apply(lambda x: np.tanh(x/p)).hist()
plt.show()


