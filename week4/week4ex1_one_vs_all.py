# used for manipulating directory paths
import os

import numpy as np;
import utils
from matplotlib import pyplot as plt
import scipy.optimize as optimize
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

class Week4Ex1():
  
  def __init__(self):
    np.random.seed(1)

  def lrCostFunction(self, theta, X, y, lambda_):
    #Initialize some useful values
    m = y.size
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    J = 0
    grad = np.zeros(theta.shape)
    
    h = utils.sigmoid(X.dot(theta.T))
    
    temp = theta
    temp[0] = 0
    
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(temp))
    
    grad = (1 / m) * (h - y).dot(X) 
    grad = grad + (lambda_ / m) * temp

    return J, grad
  
  def oneVsAll(self, X, y, num_labels, lambda_):
    # Some useful variables
    m, n = X.shape
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for c in np.arange(num_labels):
      initial_theta = np.zeros(n + 1)
      options = {'maxiter': 50}
      res = optimize.minimize(self.lrCostFunction, 
                              initial_theta, 
                              (X, (y == c), lambda_), 
                              jac=True, 
                              method='CG',
                              options=options) 
      all_theta[c] = res.x
        
    return all_theta

  def predictOneVsAll(self, all_theta, X):
    m = X.shape[0];
    num_labels = all_theta.shape[0]

    p = np.zeros(m)
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    p = np.argmax(utils.sigmoid(X.dot(all_theta.T)), axis = 1)
    return p

if __name__ == "__main__":
    one_vs_all = Week4Ex1();

    # 20x20 Input Images of Digits
    input_layer_size  = 400

    # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
    num_labels = 10

    #  training data stored in arrays X, y
    data = loadmat(os.path.join('training_data', 'ex3data1.mat'))
    X, y = data['X'], data['y'].ravel()

    # set the zero digit to 0, rather than its mapped 10 in this dataset
    # This is an artifact due to the fact that this dataset was used in 
    # MATLAB where there is no index 0
    y[y == 10] = 0
    m = y.size

    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]

    utils.displayData(sel)

    # test values for the parameters theta
    theta_t = np.array([-2, -1, 1, 2], dtype=float)

    # test values for the inputs
    X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

    # test values for the labels
    y_t = np.array([1, 0, 1, 0, 1])

    # test value for the regularization parameter
    lambda_t = 3

    J, grad = one_vs_all.lrCostFunction(theta_t, X_t, y_t, lambda_t)

    print('Cost : {:.6f}'.format(J))
    print('Expected cost: 2.534819')
    print('Gradients:')
    print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
    print('Expected gradients:')
    print(' [0.146561, -0.548558, 0.724722, 1.398003]');

    lambda_ = 0.1
    all_theta = one_vs_all.oneVsAll(X, y, num_labels, lambda_)

    pred = one_vs_all.predictOneVsAll(all_theta, X)
    print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))