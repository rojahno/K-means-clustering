import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame


class KMeans:

    def __init__(self, k=2, iterations=100, original_data=None):
        self.k = k
        self.z = None
        self.centroids = None
        self.iterations = iterations
        self.original_data = original_data

    def fit(self, X: DataFrame):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        lowest_distortion = None
        self.original_data = X
        for i in range(self.iterations):
            random_centroids = X.sample(n=self.k).to_numpy()  # Place the centroids randomly
            z, new_centroids = self.algorithm(X, random_centroids)
            distortion = euclidean_distortion(X, z)
            if i == 0 or distortion < lowest_distortion:
                lowest_distortion = distortion
                self.centroids = new_centroids
                self.z = z

    def algorithm(self, data_set: DataFrame, centroids: np.ndarray):
        """The K-means algorithm"""
        nr_data_points = data_set.shape[0]  # The amount of data points.
        old_centroids = centroids
        for index in range(nr_data_points):
            data_array = data_set.to_numpy()  # The data from the dataset in a ndarray
            z = self.assign_cluster(data_array, old_centroids, nr_data_points)
            new_centroids = self.set_centroid_positions(data_array, z)
            if self.is_converging(new_centroids, old_centroids):  # we have reached convergence
                break
            old_centroids = new_centroids
        return z, new_centroids

    def is_converging(self, new_centroid, old_centroid):
        """
        Check if the centroid is converging
        Args:
            new_centroid: The new centroids
            old_centroid: The old centroid

        Returns: True if the centroids are converging

        """
        converging = False
        if (new_centroid == old_centroid).all():
            converging = True
        return converging

    def assign_cluster(self, data: np.ndarray, centroids: np.ndarray, nr_data_points):
        """
        Assigns the data points to a cluster
        Args:
            data: The data from the dataset
            centroids: The centroids
            nr_data_points: The amount of data points

        Returns: An array indicating which data points belonging to which cluster.

        """
        z = np.zeros(nr_data_points, int)
        for index in range(nr_data_points):
            z[index] = np.argmin(euclidean_distance(np.array([data[index]] * self.k), centroids))
        return z

    def set_centroid_positions(self, data: np.ndarray, z):
        """
        Sets the position of the centroid based on the mean of the datapoints
        Args:
            data: The data from the dataset
            z: An array indicating which data points belinging to which cluster.

        Returns: The new centroid with their new positions.

        """
        new_centroids = np.zeros((self.k, data.shape[1]))
        for index in range(self.k):
            new_centroids[index] = (np.mean(data[z == index], axis=0))
        return new_centroids

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        return self.assign_cluster(X.to_numpy(), self.centroids, X.shape[0])

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>>#  model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.set_centroid_positions(self.original_data, self.z)


# --- Some utility functions


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points

    Args:
        x (array<m,d>): float tensor with pairs of
            n-dimensional points.
        y (array<n,d>): float tensor with pairs of
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))


def first_example():
    # Gets the dataset
    data_1 = pd.read_csv('data_1.csv')
    # Shows the dataset in a figure
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', data=data_1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.show()

    # Fit Model
    X = data_1[['x0', 'x1']]
    model_1 = KMeans()
    model_1.fit(X)

    # # Compute Silhouette Score
    z = model_1.predict(X)
    print(f'Silhouette Score: {euclidean_silhouette(X, z) :.3f}')
    print(f'Distortion: {euclidean_distortion(X, z) :.3f}')
    #
    # Plot cluster assignments
    C = model_1.get_centroids()
    K = len(C)
    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax)
    sns.scatterplot(x=C[:, 0], y=C[:, 1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove()


def second_example():
    data_2 = pd.read_csv('data_2.csv')
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x='x0', y='x1', data=data_2)
    # Fit Model
    Xc = data_2[['x0', 'x1']]
    model_2 = KMeans(k=10, original_data=Xc, iterations=50)
    X = normalize(Xc)
    model_2.fit(X)

    # Compute Silhouette Score
    z = model_2.predict(X)
    print(f'Distortion: {euclidean_distortion(X, z) :.3f}')
    print(f'Silhouette Score: {euclidean_silhouette(X, z) :.3f}')

    # Plot cluster assignments
    C = model_2.get_centroids()
    K = len(C)
    _, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.scatterplot(x='x0', y='x1', hue=z, hue_order=range(K), palette='tab10', data=X, ax=ax)
    sns.scatterplot(x=C[:, 0], y=C[:, 1], hue=range(K), palette='tab10', marker='*', s=250, edgecolor='black', ax=ax)
    ax.legend().remove()


def main():
    first_example()
    second_example()
    plt.show()


if __name__ == "__main__":
    main()
