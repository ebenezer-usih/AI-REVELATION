#HIERARCHICAL CLUSTERING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#USING THE DENDROGRAM TO FIND THE OPTIMAL NUMBER OF CLUSTERS
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward is used to minimize variance between clusters
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean distances')
plt.show()

#FITTING HIERARCHICAL CLUSTERING TO DATA
from sklearn.cluster import AgglomerativeClustering
HC = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_HC = HC.fit_predict(X)

#VISUALIZING RESULT
plt.scatter(X[y_HC == 0, 0], X[y_HC == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_HC == 1, 0], X[y_HC == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_HC == 2, 0], X[y_HC == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_HC == 3, 0], X[y_HC == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_HC == 4, 0], X[y_HC == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()