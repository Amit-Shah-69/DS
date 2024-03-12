#pract11A

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.rand(100, 2)
n_components = 1
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.8)
plt.title(f"Data after PCA ({n_components} component)") plt.show()

#pract11B

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42) X = np.random.rand(100, 5)
num_components = min(X.shape[1], 5)
explained_variances = []
for n_components in range(1, num_components + 1):
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
explained_variances.append(np.sum(pca.explained_variance_ratio_))
plt.plot(range(1, num_components + 1), explained_variances, marker='o')
plt.xlabel('Number of Components') plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True) plt.show() 

#pract11C

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.rand(100, 5) n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
plt.title(f'Reduced-dimensional Space (First {n_components} Components)')
plt.xlabel(f'Principal Component 1 (Explained Variance: 
{pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'Principal Component 2 (Explained Variance: 
{pca.explained_variance_ratio_[1]:.2f})')
plt.grid(True) plt.show()
