import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
plt.show()

print(X.shape, y.shape)
print("matrice X :\n", X)