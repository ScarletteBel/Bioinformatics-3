
import pandas as pd 
import matplotlib.pyplot as plt


df_cancer = pd.read_csv('surveylungcancer.csv')
print(df_cancer)

# yes= 2 , no= 1
data_types = df_cancer.dtypes 
print(data_types)

descr = df_cancer.describe()
print(descr)


plt.rcParams["figure.figsize"] = [7.5, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

df_cancer['AGE'].value_counts().plot(ax=ax, kind='bar', xlabel= 'Age', ylabel= 'Frequency')
plt.show()