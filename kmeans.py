#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:].values

#using elbow method get optimal no. of clusters
from sklearn.cluster import KMeans
wcss = []
#calculate the value for each i and plot
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('wcss values')
plt.xlabel('No. of clusters')
plt.ylabel('wcss value')
plt.show()

#training the model after getting optimal clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

#model ready
#code for visualization when only 2 features is taken
"""
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
"""