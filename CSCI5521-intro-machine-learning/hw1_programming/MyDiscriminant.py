import numpy as np


def compute_mean(self, Xtrain, ytrain, class_i):
    self.total = 0

    def inclass(x):
        if x == class_i:
            self.total += 1
            return 1
        else:
            return 0

    in_class = list(map(inclass, ytrain))
    return np.dot(in_class, Xtrain) / self.total


def compute_covariance(self, Xtrain, ytrain, mean, d):

    def rt(c, i):  # only add values to the covariance matrix if the row is in the class
        if c == ytrain[i]:
            self.count[c - 1] += 1
            return 1
        else:
            return 0

    s1 = np.zeros((d, d))
    s2 = np.zeros((d, d))

    for t in range(len(ytrain)):
        s1 = s1 + (np.outer(np.transpose(Xtrain[t] - mean[0]), np.array(Xtrain[t] - mean[0]))) * (rt(1, t))
        s2 = s2 + (np.outer(np.transpose(Xtrain[t] - mean[1]), np.array(Xtrain[t] - mean[1]))) * (rt(2, t))

    return s1 / self.count[0], s2 / self.count[1]


class GaussianDiscriminant:
    """
    Multivariate Gaussian classifier assumining class-dependent covariance
    """

    def __init__(self, k=2, d=8, priors=None):  # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k, d))  # mean
        self.S = np.zeros((k, d, d))  # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0 / k for i in range(k)]  # assume equal priors if not given
        self.k = k
        self.d = d
        self.count = [0, 0]

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        self.mean[0] = compute_mean(self, Xtrain, ytrain, 1)
        self.mean[1] = compute_mean(self, Xtrain, ytrain, 2)
        # compute the class-dependent covariance
        self.S[0], self.S[1] = compute_covariance(self, Xtrain, ytrain, self.mean, self.d)

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0])  # placeholder
        desc = [0, 0]

        for i in np.arange(Xtest.shape[0]):  # for each test set example
            for c in np.arange(self.k):  # calculate discriminant function value for each class
                desc[c] = -0.5 * (np.log(np.linalg.det(self.S[c]))) \
                    - 0.5 * np.dot(np.dot((Xtest[i] - self.mean[c]), np.linalg.inv(self.S[c])),
                                        np.transpose(Xtest[i] - self.mean[c])) + np.log(self.p[c])
            if desc[0] > desc[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S[0, :, :], self.S[1, :, :]


class GaussianDiscriminant_Ind:
    """
    Multivariate Gaussian classifier assumining class-independent covariance
    """

    def __init__(self, k=2, d=8, priors=None):  # k is number of classes, d is number of features
        # k and d are needed to initialize mean and covariance matrices
        self.mean = np.zeros((k, d))  # mean
        self.S = np.zeros((d, d))  # class-independent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0 / k for i in range(k)]  # assume equal priors if not given
        self.k = k
        self.d = d
        self.count = [0, 0]

    def fit(self, Xtrain, ytrain):

        self.mean[0] = compute_mean(self, Xtrain, ytrain, 1)
        self.mean[1] = compute_mean(self, Xtrain, ytrain, 2)
        # compute the class-independent covariance

        s1, s2 = compute_covariance(self, Xtrain, ytrain, self.mean, self.d)
        # combine the two covariances by multiplying each by their priors using a diagonal matrix and adding them together
        self.S = np.dot(s1, np.diag(np.full(self.d, self.p[0]))) + np.dot(s2, np.diag(np.full(self.d, self.p[1])))

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0])  # placeholder
        desc = [0, 0]
        for i in np.arange(Xtest.shape[0]):  # for each test set example
            for c in np.arange(self.k):  # calculate discriminant function value for each class
                desc[c] = - 0.5 * np.dot(np.dot((Xtest[i] - self.mean[c]), np.linalg.inv(self.S)),
                               np.transpose(Xtest[i] - self.mean[c])) + np.log(self.p[c])

            if desc[0] > desc[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
            # determine the predicted class based on the discriminant function values
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
