
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


df_cancer = pd.read_csv('surveylungcancer.csv')
print(df_cancer)



# yes= 2 , no= 1
data_types = df_cancer.dtypes 
print(data_types)

descr = df_cancer.describe()
print(descr, '\n')


plt.rcParams["figure.figsize"] = [7.5, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

df_cancer['AGE'].value_counts().plot(ax=ax, kind='bar', xlabel= 'Age', ylabel= 'Frequency')
plt.show()


print(df_cancer['AGE'].mean())
print(df_cancer['AGE'].median())

df_cancer['AGE'].plot.hist(bins=20)
plt.show()


sns.displot(df_cancer, x= 'AGE', hue= 'LUNG_CANCER', multiple= 'stack')
plt.show()

df_cancer21  =  df_cancer[(df_cancer["AGE"]== 21) & (df_cancer["LUNG_CANCER"] == "YES")]
print(df_cancer21)

df_cancer22  =  df_cancer[(df_cancer["AGE"]== 21) & (df_cancer["LUNG_CANCER"] == "NO")]
print(df_cancer22)