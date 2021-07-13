
"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. final weight vector w
    2. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyLeastSquare(X,y) function.

"""

# Header
import numpy as np

# Solve the least square problem by setting gradients w.r.t. weights to zeros
def GetPrediction(X, w):
    return 1 if np.matmul(X, w) > 0 else -1

def GetErrorRate(X, w, y):
    predictions = []
    incorrect_predictions = 0

    for p in range(len(X) - 1):
        prediction = GetPrediction(X[p], w)
        if prediction != y[p]:
            incorrect_predictions += 1
        predictions.append(prediction)
    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples
    return incorrect_predictions / len(X)

def MyLeastSquare(X, y):

    # calculate the optimal weights based on the solution of Question 1
    w = np.matmul(np.matmul(np.transpose(X), y), (np.power((np.matmul(np.transpose(X), X)), -1)))

    # compute the error rate
    # error rate = ( number of prediction ! = y ) / total number of training examples
    error_rate = GetErrorRate(X, w, y)

    return w, error_rate
