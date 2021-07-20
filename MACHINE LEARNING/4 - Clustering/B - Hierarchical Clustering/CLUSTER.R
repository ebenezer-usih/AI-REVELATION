#HIERARCHICAL CLUSTERING

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X = dataset[4:5]

#USING THE DENDROGRAM TO FIND THE OPTIMAL NUMBER OF CLUSTERS
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

#FITTING HIERARCHICAL CLUSTERING TO DATA
HC = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_HC = cutree(HC, 5)

# Visualising the clusters
library(cluster)
clusplot(dataset,
         y_HC,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')
