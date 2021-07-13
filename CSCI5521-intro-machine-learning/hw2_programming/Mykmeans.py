#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=6): # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 1000, 1001, 1500, 2000]
        self.center = X[init_idx]
        num_iter = 0 # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False


        # iteratively update the centers of clusters till convergence
        while is_converged == False:
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                distances = []
                # use euclidean distance to measure the distance between sample and cluster centers
                for j in range(self.num_cluster):
                    distances.append(np.linalg.norm(X[i] - self.center[j]))
                
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                cluster_assignment[i] = np.argmin(distances)

            # update the centers based on cluster assignment (M step)
            for k in range(len(self.center)):
                self.center[k] = X[cluster_assignment==k].mean(axis=0)
                

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1
        # construct the contingency matrix
        """
        The contingency matrix represents the distribution of classes for each cluster. Suppose
        that we have 3 clusters and 300 samples (100 samples per class),
        and the algorithm generates perfect clusters (samples from different classes assigned
        to different clusters), the matrix could look like:
        [[100,0,0],
        [0,100,0],
        [0,0,100]]
        where each row corresponds to a cluster and each column for a class (in the order of digit 0,8,9
        for this assignment). Ideally, we would like the majority of samples in a cluster belong to a single
        class.
        """
        contingency_matrix = np.zeros([self.num_cluster,3])
        for i in range(len(X)):
            if y[i] == 0:
                contingency_matrix[cluster_assignment[i]][0] += 1
            elif y[i] == 8:
                contingency_matrix[cluster_assignment[i]][1] += 1
            else:
                contingency_matrix[cluster_assignment[i]][2] += 1
        return num_iter, self.error_history, contingency_matrix

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error += np.linalg.norm(X[i] - self.center[cluster_assignment[i]])
        return error

    def params(self):
        return self.center
