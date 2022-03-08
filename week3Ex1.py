from mimetypes import init
import numpy as np;
import utils
from matplotlib import pyplot as plt
import scipy.optimize as optimize

class Week3Ex1():
  
  def __init__(self):
     np.random.seed(1)
  
  def plotData(self, x, y):
    fig = plt.figure()

    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
  
  def sigmoid(self, z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.exp(-z))
    return g
  
  def costFunction(self, theta, X, y):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    h = self.sigmoid(X.dot(theta.T))
    
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))
    grad = (1 / m) * (h - y).dot(X)
    return J, grad

  def predict(self, theta, X):
    m = X.shape[0]
    p = np.zeros(m)

    p = np.round(self.sigmoid(X.dot(theta.T)))
    return p

if __name__ == "__main__":
    logistic_regression = Week3Ex1();

    data = np.loadtxt('training_data/ex2data1.txt', delimiter=',');
    X, y = data[:, 0:2], data[:, 2]
    [m, n] = X.shape

    # % Print out some data points
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    
    for i in range(10):
      print(X[i, 0], X[i, 1], y[i])

    logistic_regression.plotData(X, y)
    # add axes labels
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    pass

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    z = 0
    g = logistic_regression.sigmoid(z)

    initial_theta = np.zeros(n + 1);
    cost, grad = logistic_regression.costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx): 0.693\n')

    print('Gradient at initial theta (zeros):')
    print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost, grad = logistic_regression.costFunction(test_theta, X, y)

    print('Cost at test theta: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.218\n')

    print('Gradient at test theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

    # optimize.minimize will converge on the right optimization parameters 
    # and return the final values of the cost and in a class object. 
    # Notice that by using optimize.minimize, 
    # you did not have to write any loops yourself, 
    # or set a learning rate like you did for gradient descent.

    # set options for optimize.minimize
    options= {'maxiter': 400}

    res = optimize.minimize(logistic_regression.costFunction,
                            initial_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun
    # the optimized theta is in the x property
    theta = res.x

    # Print theta to screen
    print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.203\n');

    print('theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
    print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

    # Plot Boundary
    utils.plotDecisionBoundary(logistic_regression.plotData, theta, X, y)
    # plt.show()

    #  Predict probability for a student with score 45 on exam 1 
    #  and score 85 on exam 2 
    prob = logistic_regression.sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85,'
          'we predict an admission probability of {:.3f}'.format(prob))
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = logistic_regression.predict(theta, X)
    print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.00 %')