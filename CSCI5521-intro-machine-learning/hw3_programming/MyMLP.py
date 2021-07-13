import numpy as np


def process_data(data, mean=None, std=None):
    # normalize the data to have zero mean and unit variance (add 1xe-15 to std to avoid numerical issue)
    if mean is not None:
        # mean and std is precomputed with the training data
        return (data - mean) / std
    else:
        # compute the mean and std based on the training data
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-15
        data = (data - mean) / std
        return data, mean, std


def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    for i in range(len(label)):
        one_hot[i][label[i]] = 1
    return one_hot


def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    return 1/(1 + np.exp(-x))


def softmax(x):
    # implement the softmax activation function for output layer
    return np.exp(x) / np.sum(np.exp(x))


class MLP:
    def __init__(self, num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64, num_hid])  # w
        self.bias_1 = np.random.random([1, num_hid])
        self.weight_2 = np.random.random([num_hid, 10])  # v
        self.bias_2 = np.random.random([1, 10])
        self.num_hid = num_hid

    def fit(self, train_x, train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 100 iterations
        """
        while count <= 100:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass
            z = self.zero_node(self.get_hidden(train_x))
            v = np.vstack((self.bias_2, self.weight_2))
            y = np.array([softmax(z_i) for z_i in z@v])

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            delta_v = lr * ((train_y - y).T@z).T
            delta_w = lr * (((train_y - y)@v.T * z *
                             (np.ones((len(train_x), self.num_hid + 1)) - z)).T@self.zero_node(train_x)).T

            # update the parameters based on sum of gradients for all training samples
            self.weight_1 = self.weight_1 + delta_w[1:, 1:]
            self.bias_1 = self.bias_1 + delta_w[0, 1:]
            self.weight_2 = self.weight_2 + delta_v[1:, :]
            self.bias_2 = self.bias_2 + delta_v[0, :]

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(
                predictions.reshape(-1) == valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1
        return best_valid_acc

    def predict(self, x):
        # generate the predicted probability of different classes
        # convert class probability to predicted labels
        z = self.zero_node(self.get_hidden(x))
        v = np.vstack((self.bias_2, self.weight_2))
        probabilities = np.array([softmax(z_i) for z_i in z@v])
        return np.array([np.argmax(y_i) for y_i in probabilities])

    def get_hidden(self, x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        w = np.vstack((self.bias_1, self.weight_1))
        return np.array([sigmoid(x_i) for x_i in self.zero_node(x)@w])

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2

    def zero_node(self, x):
        return np.hstack((np.ones(len(x))[..., None], x))
