import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.2
iters = 10

# reading the dataset
df = pd.read_csv('../Datasets/EV.csv')
df = df[['ELAPSED_TIME', 'DISTANCE']].dropna(how='any')

X = df.iloc[:, 1:].values
Y = df.iloc[:, 0].values

# normalization
X = X / X.max(axis=0)
Y = Y / Y.max(axis=0)

# adding the extra column of ones for the bias
X = np.append(np.ones((len(X), 1)), X, axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# generating the random weights
w = np.random.normal(size=2)

# initialize error history array
errorHistory = []

# training the preceptron
for j in range(iters):
    errors = 0
    for i, row in enumerate(X_train):
        dot = np.dot(row, w)
        term = learning_rate * (dot - Y_train[i]) * row
        w -= term
        errors += (dot - Y_train[i]) ** 2
    errorHistory.append(errors / len(X_train))

# predict the test data
predictions = np.dot(X_test, w)
errors = 0
for i in range(len(X_test)):
    errors += (predictions[i] - Y_test[i]) ** 2
errors /= len(X_test)
print("Mean square error over test set:", errors)

# plotting the dataset entries 
plt.scatter([X_test[i][1] for i in range(len(X_test))], Y_test, color = 'blue')
plt.plot([X_test[i][1] for i in range(len(X_test))], np.dot(X_test, w), color='red')

# setting title and labels
plt.title('Time VS Distance')
plt.xlabel('Distance')
plt.ylabel('Time')

# displaying the plot
plt.show()
 
# plotting mean error vs iterations
plt.figure(2)
plt.plot(np.arange(0, iters), errorHistory)

# setting title and labels
plt.title('Number of iterations VS Mean Squared error')
plt.xlabel('Number of iterations')
plt.ylabel('Mean Squared error')

# displaying the plot
plt.show()
