import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.2
iters = 200

# reading the dataset
df = pd.read_csv('../Datasets/m_creditcard_24650.csv')
X = df.iloc[:, 3:-1].values
Y = df.iloc[:, -1].values

# adding the extra column of ones for the bias
X = np.append(np.ones((len(X), 1)), X, axis=1)

# generating the random weights
w = np.random.normal(size=3)

# initialize error history array
errorHistory = []

# training the preceptron
for j in range(iters):
    errors = 0
    for i, row in enumerate(X):
        dot = np.dot(w, row)
        activation = 1 if dot > 0 else 0
        term = learning_rate * (activation - Y[i]) * row
        w -= term
        errors += 1 if (activation != Y[i]) else 0
    errorHistory.append(errors / len(X))

# plotting the dataset entries 
plt.scatter([X[i][1] for i in range(len(X)) if(Y[i] == 1)], [X[i][2] for i in range(len(X)) if(Y[i] == 1)], color = 'blue', marker='o')
plt.scatter([X[i][1] for i in range(len(X)) if(Y[i] == 0)], [X[i][2] for i in range(len(X)) if(Y[i] == 0)], color = 'red', marker='x')

# plotting the classification line
ly = np.arange(-5, 15)
lx = (-w[0] - w[2] * ly) / w[1]
plt.plot(lx, ly)

# setting title and labels
plt.title('V11 vs V2')
plt.xlabel('V2')
plt.ylabel('V11')

# displaying the plot
plt.show()
 
# plotting mean error vs iterations
plt.figure(2)
plt.plot(np.arange(0, iters), errorHistory)

# setting title and labels
plt.title('Number of iterations VS Number of errors')
plt.xlabel('Number of iterations')
plt.ylabel('Number of errors')

# displaying the plot
plt.show()
