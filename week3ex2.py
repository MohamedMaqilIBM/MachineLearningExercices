from mimetypes import init
import numpy as np;
import utils
from matplotlib import pyplot as plt
import scipy.optimize as optimize

class week3Ex2():
  
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

  def costFunctionReg(self, theta, X, y, lambda_):
    m = y.size

    J = 0
    grad = np.zeros(theta.shape)

    h = self.sigmoid(X.dot(theta.T))
    
    temp = theta
    temp[0] = 0
    
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(temp))

    grad = (1 / m) * (h - y).dot(X) 
    grad = grad + (lambda_ / m) * temp

    return J, grad

  
  def predict(self, theta, X):
    m = X.shape[0]
    p = np.zeros(m)

    p = np.round(self.sigmoid(X.dot(theta.T)))
    return p

if __name__ == "__main__":
    regularized_logistic_regression = week3Ex2();

    data = np.loadtxt('ex2data2.txt', delimiter=',');
    X, y = data[:, :2], data[:, 2]
    [m, n] = X.shape

    regularized_logistic_regression.plotData(X, y)
    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    # Specified in plot order
    plt.legend(['y = 1', 'y = 0'], loc='upper right')
    # plt.show()
    # pass

    # Note that mapFeature also adds a column of ones for us, so the intercept term is handled
    X = utils.mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    # DO NOT use `lambda` as a variable name in python
    # because it is a python keyword
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = regularized_logistic_regression.costFunctionReg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx)       : 0.693\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])
    cost, grad = regularized_logistic_regression.costFunctionReg(test_theta, X, y, 10)

    print('------------------------------\n')
    print('Cost at test theta    : {:.2f}'.format(cost))
    print('Expected cost (approx): 3.16\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')


    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1 (you should vary this)
    lambda_ = 1.00

    # set options for optimize.minimize
    options= {'maxiter': 100}

    res = optimize.minimize(regularized_logistic_regression.costFunctionReg,
                            initial_theta,
                            (X, y, lambda_),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of OptimizeResult object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property of the result
    theta = res.x

    utils.plotDecisionBoundary(regularized_logistic_regression.plotData, theta, X, y)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0'])
    plt.grid(False)
    plt.title('lambda = %0.2f' % lambda_)

    # Compute accuracy on our training set
    p = regularized_logistic_regression.predict(theta, X)

    print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
    print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')
