
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 




df_cancer = pd.read_csv('surveylungcancer.csv')
print(df_cancer)


# Data types description.....
# yes= 2 , no= 1
data_types = df_cancer.dtypes 
print(data_types)

#statistical data description....
descr = df_cancer.describe()
print(descr, '\n')



#### measures of central tendency  ####

# Frequency &...
plt.rcParams["figure.figsize"] = [7.5, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()

df_cancer['AGE'].value_counts().plot(ax=ax, kind='bar', xlabel= 'Age', ylabel= 'Frequency')
plt.show()

print(df_cancer['AGE'].mean())
print(df_cancer['AGE'].median())


# Histograms...
df_cancer['AGE'].plot.hist(bins=20)
plt.show()

sns.displot(df_cancer, x= 'AGE', hue= 'LUNG_CANCER', multiple= 'stack')
plt.show()


#Filtering...
df_cancer22  =  df_cancer[(df_cancer["AGE"]== 21) & (df_cancer["LUNG_CANCER"] == "NO")]
print(df_cancer22)





#Correlation matriz...
corr_matrix = df_cancer.corr()
print(corr_matrix)

sns.heatmap(corr_matrix)
plt.show()



#### measures of disepersion ####

standar_Dev = df_cancer['AGE'].std()
print(standar_Dev)


rango = df_cancer['AGE'].max() - df_cancer['AGE'].min()
print(rango)



median = df_cancer['AGE'].median()

q1 = df_cancer['AGE'].quantile(q=0.25)
q3 = df_cancer['AGE'].quantile(q=0.75)

min_val = df_cancer['AGE'].quantile(q=0)
max_val = df_cancer['AGE'].quantile(q=1)

print()
print(median)
print(q1)
print(q3)
print(min_val)
print(max_val)
print()

iqr = q3 - q1
print(iqr)

#applicable for simetric distribution ..
minlimit = q1 - 1.5*iqr
maxlimit = q3 +1.5*iqr

print(minlimit, maxlimit)


sns.histplot(df_cancer['AGE'])
plt.show()

sns.boxplot(df_cancer['AGE'])
plt.show()


sns.boxplot(x= 'AGE', y='LUNG_CANCER', data= df_cancer)
plt.show()

##