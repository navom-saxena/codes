import matplotlib.pyplot as plt
import pandas as pd

# preprocessing

# reading dataset
dataset = pd.read_csv(
    "/Users/navomsaxena/codes/src/main/resources/machinelearning/clustering.csv")
print(dataset)

# extracting age and score to make clusters based on age and score
X = dataset.iloc[:, [3, 4]].values

# using elbow method to find k, no of clusters to make
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('no of clusters')
plt.show()

# k comes out to be 5 from elbow method graph
# applying k-means to dataset with k = 5
# y_kmeans gives 0 to 4 numbers as clusters referring to same row in X
k_means = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = k_means.fit_predict(X)

# visualizing clusters, making cluster one by one, pulling only those values from X where y_kmeans = cluster no
# then making centroids
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('K-means cluster')
plt.xlabel('income')
plt.ylabel('score')
plt.legend()
plt.show()
