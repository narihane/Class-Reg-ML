# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 22:52:43 2017

@author: Kareem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rate = 0.8

df = pd.read_csv('TestResults.csv')
X = df.iloc[:, 1:3].values
Y = df.iloc[:, -1].values
for i in range(len(Y)):
    Y[i] = 1 if Y[i] == '\'P\'' else -1

X = np.append(np.ones((len(X), 1)), X, axis=1)

w = np.random.normal(size=3)

errorHistory = []

for j in range(100):
    errors = 0;
    for i, row in enumerate(X):
        dot = np.dot(w, row)
        activation = 1 if dot > 0 else -1
        term = rate * (activation - Y[i]) * row
        w -= term
        errors += 1 if (activation != Y[i]) else 0
    errorHistory.append(errors)
    
print(w)
print(errorHistory)

print(1 if np.dot(np.array([1, 5, 5]), w) > 0 else -1)
    
lx = np.arange(0, 10)
ly = (-w[0] - w[1] * lx) / w[2]

plt.scatter([X[i][1] for i in range(len(X)) if(Y[i] == 1)], [X[i][2] for i in range(len(X)) if(Y[i] == 1)], color = 'blue', marker='o')
plt.scatter([X[i][1] for i in range(len(X)) if(Y[i] == -1)], [X[i][2] for i in range(len(X)) if(Y[i] == -1)], color = 'red', marker='x')
plt.plot(lx, ly)
plt.show()


