import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


class week2Ex2():
  
  def __init__(self):
     np.random.seed(1)
  
  def plotData(self, x, y):

    fig = plt.figure()  # open a new figure
    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')

  
  def computeCostMulti(self, X, y, theta):
    m = y.shape[0];
    J = 0

    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

  def gradientDescentMulti(self, X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
      theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
      J_history.append(self.computeCostMulti(X, y, theta))
  
    return theta, J_history

  def featureNormalize(self, X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    m = X.shape[0]

    # mu = np.mean(X, axis = 0)
    mu[0] = np.average(X[:, :1]);
    mu[1] = np.average(X[:, 1:2]);
    sigma = np.std(X, axis = 0)

    X_norm = (X - mu) / sigma;
    return X_norm, mu,sigma;

  def normalEqn(self, X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return theta


if __name__ == "__main__":
    feature_normalize = week2Ex2();

    data = np.loadtxt('exercices/ex1data2.txt', delimiter=',');
    X = data[:, :2]
    y = data[:, 2]
    m = y.size

    # % Print out some data points
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    
    for i in range(10):
      print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))
    
    X_norm, mu, sigma = feature_normalize.featureNormalize(X);

    print('Computed mean:', mu)
    print('Computed standard deviation:', sigma)

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    alpha = 0.1
    num_iters = 400

    theta = np.zeros(3)
    theta, J_history = feature_normalize.gradientDescentMulti(X, y, theta, alpha, num_iters)

    plt.plot(np.arange(len(J_history)), J_history, lw=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()
    pass

    print('theta computed from gradient descent: {:s}'.format(str(theta)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    X_array = [1, 1650, 3]
    X_array[1:3] = (X_array[1:3] - mu) / sigma
    price = np.dot(X_array, theta)   # You should change this

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))


    # Normal Equations
    
    # Load data
    data = np.loadtxt('ex1data2.txt', delimiter=',');
    X = data[:, :2]
    y = data[:, 2]
    m = y.size
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Calculate the parameters from the normal equation
    theta = feature_normalize.normalEqn(X, y);
    print('Theta computed from the normal equations: {:s}'.format(str(theta)));

    X_array = [1, 1650, 3]
    X_array[1:3] = (X_array[1:3] - mu) / sigma
    price = np.dot(X_array, theta)

    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))




