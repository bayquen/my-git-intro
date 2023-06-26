from __future__ import print_function, division
from builtins import range

import numpy as np

N = 100
D = 2

X = np.random.randn(N, D)       # Creates N x D normally-distributed matrix!
# We need to add a bias term. For this approach, we add a col. of 1's into
# our original data, and include the bias term in the weights, w.

# ones = np.array([[1] * N]).T  # old!
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
print(sigmoid(z))