#pract10B

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, 
edgecolors='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1') plt.ylabel('Feature 2') plt.show()

#pract10B

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
data, _ = make_blobs(n_samples=300, centers=4, random_state=42)
distortions = []
silhouette_scores = []
for k in range(2, 10):
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(data)
distortions.append(kmeans.inertia_)
silhouette_scores.append(silhouette_score(data, kmeans.labels_))
def present_findings():
overall_sales = calculate_overall_sales()
print("Finding 1: Overall Sales Trend")
print("The overall sales trend indicates a steady increase over the five time periods.")
print()
print("Finding 2: Comparison of Sales Across Categories")
print("The bar chart illustrates that Electronics has consistently higher sales, followed by Clothing.")
print()
print("Finding 3: Percentage Distribution of Sales Across Categories")for category, percentage in zip(categories, overall_sales):
print(f"{category}: {percentage:.1f}%")
print("The pie chart highlights the percentage distribution of sales across categories.")
print()
present_findings() 



