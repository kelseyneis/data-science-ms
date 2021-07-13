import numpy as np
from MyDiscriminant import GaussianDiscriminant, GaussianDiscriminant_Ind

# load data
df = np.genfromtxt("training_data.txt", delimiter=",")
dftest = np.genfromtxt("test_data.txt", delimiter=",")
Xtrain = df[:, 0:8]
ytrain = df[:, 8]
Xtest = dftest[:, 0:8]
ytest = dftest[:, 8]

# define the model with a Gaussian Discriminant function (class-dependent covariance)
clf = GaussianDiscriminant(2, 8, [0.1, 0.9])

# update the model based on training data
clf.fit(Xtrain,ytrain)

# evaluate on test data
predictions = clf.predict(Xtest)
confusion_matrix = np.array([[sum((ytest==1) & (predictions==1)),sum((ytest==2) & (predictions==1))],
                           [sum((ytest==1) & (predictions==2)),sum((ytest==2) & (predictions==2))]])
print('Confusion Matrix for Gaussian Discriminant with class-dependent covariance')
print(confusion_matrix)

# define the model with a Gaussian Discriminant function (class-independent covariance)
clf = GaussianDiscriminant_Ind(2,8,[0.1,0.9])

# update the model based on training data
clf.fit(Xtrain,ytrain)

# evaluate on test data
predictions = clf.predict(Xtest)
confusion_matrix = np.array([[sum((ytest==1) & (predictions==1)),sum((ytest==2) & (predictions==1))],
                           [sum((ytest==1) & (predictions==2)),sum((ytest==2) & (predictions==2))]])
print('Confusion Matrix for Gaussian Discriminant with class-independent covariance')
print(confusion_matrix)
