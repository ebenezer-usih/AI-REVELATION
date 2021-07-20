#K MEANS CLUSTER

#IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING DATASET
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#USING THE ELBOW METHOD TO FIND OPTIMAL NUMBER OF CLUSTERS
from sklearn.cluster import KMeans
wcss = []  #THE SQUARE BRACKETS SHOW THAT IT IS A NULL VARIABLE
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #WCSS IS ALSO CALLED INERTIA
plt.plot(range(1, 11), wcss)
plt.title('THE ELBOW METHOD')
plt.ylabel('WCSS')
plt.xlabel('NUMBER OF CLUSTERS')
plt.show()

#APPLYING KMEANS TO THE DATASET
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(X)

#VISUALISING CLUSTERS
plt.scatter(X[ykmeans == 0, 0], X[ykmeans == 0, 1], s = 100, c = 'red', label = 'cluster 1' )
plt.scatter(X[ykmeans == 1, 0], X[ykmeans == 1, 1], s = 100, c = 'blue', label = 'cluster 2' )
plt.scatter(X[ykmeans == 2, 0], X[ykmeans == 2, 1], s = 100, c = 'magenta', label = 'cluster 3' )
plt.scatter(X[ykmeans == 3, 0], X[ykmeans == 3, 1], s = 100, c = 'cyan', label = 'cluster 4' )
plt.scatter(X[ykmeans == 4, 0], X[ykmeans == 4, 1], s = 100, c = 'green', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()