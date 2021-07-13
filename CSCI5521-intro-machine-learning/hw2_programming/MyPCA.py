import numpy as np

def PCA(X,num_dim=None):
    
    # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)

    centered_data = X - np.mean(X, axis=0, keepdims=True)
    feature_covariance = centered_data.T@centered_data / (len(X) - 1)
    eigen_v = np.linalg.eigh(feature_covariance)
    eigen_values = np.flip(eigen_v[0], 0)
    eigen_vectors = np.flip(eigen_v[1], 1)
    # select the reduced dimensions that keep >90% of the variance (slide 23)
    if num_dim is None:
        next_pov = 1.0
        num_dim = len(eigen_values)
        while True:
            num_dim -= 1
            next_pov = round(eigen_values[0: num_dim].sum() / eigen_values.sum(), 3)
            if next_pov < .9:
                num_dim += 1
                break

    X_pca = centered_data@eigen_vectors[:, 0:num_dim]

    return X_pca, num_dim
