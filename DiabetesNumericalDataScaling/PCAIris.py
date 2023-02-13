import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

iris = sns.load_dataset('iris')
print(iris)

scaler = StandardScaler()
scaled = scaler.fit_transform(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values)

covariance_matrix = np.cov(scaled.T)
print(covariance_matrix)

sns.pairplot(iris)
plt.show()

sns.jointplot(x= iris['petal_length'], y=iris['petal_width'])
sns.jointplot(x=scaled[:,2], y=scaled[:,3]) #shows data asociated to an unit variance 
plt.show()


eigen_values, eigen_vectors =np.linalg.eig(covariance_matrix)
print(eigen_values)
print(eigen_vectors) #Both capture the most variance of data 


variance_explained = []
for i in eigen_values:
    variance_explained.append((i/sum(eigen_values))*100)

print()
print(variance_explained)


#Applying PCA 

pca = PCA(n_components=2)
pca.fit(scaled)

pca_explained = pca.explained_variance_ratio_
print(pca_explained)


reduced_scaled = pca.transform(scaled)
print(reduced_scaled)


iris['pca_1'] = scaled[:,0]
iris['pca_2'] = scaled[:,1]

print(iris)


