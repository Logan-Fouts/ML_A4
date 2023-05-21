import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def compute_gradient(Y, X):
    """
    Calculates the gradient for a given sammon mapping.

    Arguments:
        Y (numpy array): Config of points.
        data (numpy array): Original points.

    Return:
        numpy array: Gradient of sammon mapping.
    """
    s = Y.shape[0]
    distY = np.sqrt(np.sum((Y[:, np.newaxis] - Y) ** 2, axis=2))
    distD = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))
    grade = np.zeros(Y.shape)
    
    for i in range(s):
        tmpY = Y[i] - Y
        dataChange = X[i] - X
        tmpY[i] = 0
        dataChange[i] = 0
        # Be careful with division by zero.
        deriveE = (distY[i] - distD[i]) / ((distY[i] * distD[i]) + 1e-8)
        grade[i] = np.sum(tmpY * deriveE[:, np.newaxis], axis=0)

    return (grade * (2 / s))

def compute_stress(Y, X):
    """
    Computes stress for given sammon mapping.

    Arguments:
        Y (numpy array): Config of points.
        data (numpy array): Original points.

    Return:
        float: Current stress of sammon mapping.
    """
    distY = np.linalg.norm(Y[:, np.newaxis] - Y, axis=2)
    distD = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    stress = np.mean((distY - distD) ** 2)

    return stress

def sammon(X, iter=500, e=1e-7, a=.1):
    """
    Implements sammon mapping given a set of data and optionally 3 other parameters.

    Arguments:
        X (numpy array): Data to reduce, as an n x p matrix (n observations by p features).
        iter (int): Maximum number of iterations.
        e (float): Error threshold.
        a (float): Learning rate.
    
    Return:
        numpy array: A n x 2 vector with the final two-dimensional layout.
    """
    # 1. Start with a random two-dimensional layout Y of points (Y is a n × 2 matrix).
    Y = np.random.rand(X.shape[0], 2)

    # 2. Compute the stress E of Y.
    E = compute_stress(Y, X)

    # 3. If E < epsilon, or if the maximum number of iterations iterations has been reached, stop.
    for i in range(iter):
        if E < e:
            break
    # 4. For each yi of Y, find the next vector yi(t + 1) based on the current yi(t).
        gradient = compute_gradient(Y, X)
        Y -= a * gradient
        E = compute_stress(Y, X)
    
    return Y

################################ Testing ################################
iris = datasets.load_iris()
(x,index) = np.unique(iris.data,axis=0,return_index=True)
target = iris.target[index]
names = iris.target_names
np.random.seed(0)

iter = 1000
e = 1e-7
a = 1
y = sammon(x, iter, e, a)

plt.scatter(y[target ==0, 0], y[target ==0, 1], s=20, c='r', marker='o',label=names[0])
plt.scatter(y[target ==1, 0], y[target ==1, 1], s=20, c='b', marker='D',label=names[1])
plt.scatter(y[target ==2, 0], y[target ==2, 1], s=20, c='y', marker='v',label=names[2])
plt.show()