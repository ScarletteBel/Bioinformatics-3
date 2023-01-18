# Final Project   Scarlette Bello       c0860234


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix



##  DATASET CONTEXT 
##  The dataset called 'physiology' is used in this project. It is a dataset obtained from biological scientific arcticles to study...
##  the behavior of a certain scpecie of bird when it is near of migrate, relating this with their physical features
##  The database used in this project was obtain from PubChem Database, a scientific public database 

data = pd.read_csv('physiology.csv')
print(data,'\n')
print(data.shape)

empty = data.isnull().sum()
print(empty)



## TARGET VARIABLES 
# The "days_to_departure" data is chosen to analyse in this project, it is the target variable since the analysis..
# is to know if the bird's phisiology affects the imigration departure.
# The "days_to_departure" data is continuous and numerical.

# The Independent variables are 'Muscle' and 'Mass', the phisiology information. 
# These data are also numerical.

x=data.loc[:,['Mass']]
y=data.loc[:,'days_to_departure']

xshape = x.shape
yshape = y.shape 

print(y)
print(x)

print(xshape)
print(yshape)



## DATA PREPROCESSING...

#There are no missing values in this data set, usually in this kind of data there are.
#The data does not need to be processed. The information is numeriacal and it is possible to manipolate without previous processing.

empty = data.isnull().sum()
print(empty)



## DATA VISUALIZATION...
## A scatterplot and an histogram were used for the preanalize of the data. There is a correlation with the mass and date of departure.
## The data has a similar behaviour of gaussian distribution. The data has a good distribution for the aplication of the model since it is central and without separate values.

plt.scatter(x =data['Mass'],  y=data['days_to_departure'], color=['green'])
plt.show()

plt.hist(x, density=True, bins=30)  
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()



## APLYING MODELS 

# Linear Regression is the first model tested 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

print()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

##let's go to model the linear regression!!!
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = linear_model.LinearRegression()
trainmodel = model.fit(x_train, y_train)

y_pred = model.predict(x_test)

##Linear regression model performance verifiers...
print()
print("Coeficients:",model.coef_)
print("Intercept:",model.intercept_)
print("Mean squared error (MSE): ", mean_squared_error(y_test,y_pred))
print("Coefficient of determination (R^2): ", r2_score(y_test,y_pred))

##Making a scatter plot....
import matplotlib.pyplot as plt
import seaborn as sns 

sns.scatterplot(x = y_test,y = y_pred, alpha = 0.5)
plt.show() 




#K Nearest Neighbours is tested 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
print(x_train.shape)

model_knn =KNeighborsRegressor(n_neighbors=3)
model_knn.fit(x_train,y_train)

predict_train = model_knn.predict(x_train)
predict_test = model_knn.predict(x_test)
print('\n')


#Performance of the model on test data...
print(confusion_matrix(y_test,predict_test))
print('\n')
print(classification_report(y_test,predict_test))




#The linear regression and KNN models where used because the aim of the analysis is to know if there is a correlation between the physiology and
#   the date of inmigrations odf the species; Nevertheless, taht correlation is low.
#The parameters used in the two models are different; for the kneighbourds is necessary to stablish a value of K,
#  to know whith many near points the data has to be compared and relationated.


#The behaviour of the model is compared with the accuracy; for this dataset, the K neighbour is the model which best fits, with a nearest 
#  accuracy to the 1 value, nearest than the linear regression model. Also, the F1 parameter is nearer to 1.













