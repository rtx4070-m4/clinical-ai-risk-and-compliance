
from sklearn.decomposition import PCA

def pca_reduce(X, n_components=5):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca
