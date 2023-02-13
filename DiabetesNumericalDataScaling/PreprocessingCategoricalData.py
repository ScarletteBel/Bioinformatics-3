import pandas as pd 
import sklearn.preprocessing as preprocessing 



df = pd.read_csv('cars.csv')
print(df )

engine_dummies = pd.get_dummies(df['engine_type'])
print(engine_dummies)



encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

encoder.fit(df[['engine_type']].values)
array = encoder.transform([['gasoline'],['diesel'],['oil']]).toarray()
print(array)


