import pandas as pd
import numpy as np
# height (cm)
X = np.array([[0.8, 2.2, 2.4, 4, 5]]).T
# weight (kg)
y = np.array([[0.6, 0.5, 1, 0.3, 0.3]]).T

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((X, one), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print(w)


