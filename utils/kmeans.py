# coding=utf-8

'''
@author: wepon, http://2hwp.com
Reference:
            Book: <<Machine Learning in Action>>
            Software: sklearn.cluster.KMeans
'''
import numpy as np


class KMeans(object):
    """
    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    initCent : str or array-like
        Initialization method for centroids.
        Options: "random" or a user-specified numpy array.
        Default = "random" (random initialization).
    max_iter : int
        Maximum number of iterations.
    """

    def __init__(self, n_clusters=5, initCent='random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=float)
        else:
            self.centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None
        self.sse = None

    # Compute Euclidean distance between two points
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    # Randomly select k centroids within the data boundaries
    def _randCent(self, X, k):
        n = X.shape[1]  # feature dimension
        centroids = np.empty((k, n))  # k*n matrix for storing centroids
        for j in range(n):  # randomly initialize centroids dimension by dimension
            minJ = min(X[:, j])
            rangeJ = float(max(X[:, j]) - minJ)
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
        return centroids

    def fit(self, X):
        # Type check
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # number of samples
        # clusterAssment: (m,2) matrix
        # first column = cluster index
        # second column = squared error with assigned centroid
        self.clusterAssment = np.empty((m, 2))
        if self.initCent == 'random':
            self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for _ in range(self.max_iter):
            clusterChanged = False
            for i in range(m):  # assign each sample to the nearest centroid
                minDist = np.inf
                minIndex = -1
                for j in range(self.n_clusters):
                    distJI = self._distEclud(self.centroids[j, :], X[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if self.clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i, :] = minIndex, minDist ** 2

            if not clusterChanged:  # convergence check
                break
            for i in range(self.n_clusters):  # update centroids
                ptsInClust = X[np.nonzero(self.clusterAssment[:, 0] == i)[0]]
                self.centroids[i, :] = np.mean(ptsInClust, axis=0)

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])

    def predict(self, X):
        # Predict cluster assignment for new data
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):  # assign to nearest centroid
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds


class biKMeans(object):
    """
    Bisecting KMeans:
    Repeatedly splits clusters until the desired number of clusters is reached.
    """

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusterAssment = None
        self.labels = None
        self.sse = None

    # Compute Euclidean distance between two points
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

    def fit(self, X):
        m = X.shape[0]
        self.clusterAssment = np.zeros((m, 2))
        centroid0 = np.mean(X, axis=0).tolist()
        centList = [centroid0]
        for j in range(m):  # initial squared error for each point
            self.clusterAssment[j, 1] = self._distEclud(np.asarray(centroid0), X[j, :]) ** 2

        while len(centList) < self.n_clusters:
            lowestSSE = np.inf
            for i in range(len(centList)):  # try splitting each cluster
                ptsInCurrCluster = X[np.nonzero(self.clusterAssment[:, 0] == i)[0], :]
                clf = KMeans(n_clusters=2)
                clf.fit(ptsInCurrCluster)
                centroidMat, splitClustAss = clf.centroids, clf.clusterAssment
                sseSplit = sum(splitClustAss[:, 1])
                sseNotSplit = sum(self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] != i)[0], 1])
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            # Update cluster assignment after split
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
            centList[bestCentToSplit] = bestNewCents[0, :].tolist()
            centList.append(bestNewCents[1, :].tolist())
            self.clusterAssment[np.nonzero(self.clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss

        self.labels = self.clusterAssment[:, 0]
        self.sse = sum(self.clusterAssment[:, 1])
        self.centroids = np.asarray(centList)

    def predict(self, X):
        # Predict cluster assignment for new data
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):  # assign to nearest centroid
            minDist = np.inf
            for j in range(self.n_clusters):
                distJI = self._distEclud(self.centroids[j, :], X[i, :])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds
