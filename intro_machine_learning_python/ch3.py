from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
print(cancer.target)
X_scaled = scaler.transform(cancer.data)
print(np.shape(X_scaled))

pca = PCA(n_components=2)
# fit PCA model to breast cancer data
pca.fit(X_scaled)
#print(type(X_scaled))