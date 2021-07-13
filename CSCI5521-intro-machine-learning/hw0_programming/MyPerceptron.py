"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package

import numpy as np

def WeightHasChanged(w, prevW):
    comparison = w == prevW
    return not comparison.all()

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

# Implement the Perceptron algorithm
def MyPerceptron(X,y,w0=[1.0,-1.0]):
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w = w0
    prevW = [0.0, 0.0]
    error_rate = 1.00

    # loop until convergence (means when w does not change at all over one pass)
    # or until max iterations are reached
    # (current pass w ! = previous pass w), then do:
    #

    # make prediction on the csv dataset using the feature set
    # Note that you need to convert the raw predictions into binary predictions using threshold 0
    while WeightHasChanged(w, prevW):
        prevW = w
        # for each training sample (x,y):
        for t in range(len(X) - 1):
            # if actual target y does not match the predicted target value, update the weights
            if y[t] * np.matmul(w, X[t]) <= 0:
                w = w + y[t]*X[t]
                # calculate the number of iterations as the number of updates
                k += 1

    error_rate = GetErrorRate(X, w, y)

    return w, k, error_rate
