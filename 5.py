from sklearn.cluster import KMeans
import numpy as np
data = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(data)
print("Labels:", kmeans.labels_)
