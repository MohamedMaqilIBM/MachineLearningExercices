import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


class week2Ex1():
  
  def __init__(self):
     np.random.seed(1)
  
  def plotData(self, x, y):

    fig = plt.figure()  # open a new figure
    plt.plot(x, y, 'ro', ms=10, mec='k')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')

  def computeCost(self, X, y, theta):
    m = y.shape[0];
    predictions = np.dot(X, theta);
    sqrError = (predictions - y) ** 2;

    return np.sum(sqrError)/(2*m);

  def linearRegression(self, X,y,theta, alpha, iterations):
    m = y.shape[0];
    theta = theta.copy();
    J_history = []
    for i in range(iterations):
      gradJ = (np.dot(X, theta) - y).dot(X);
      theta = theta - (alpha / m) * gradJ;
      J_history.append(self.computeCost(X, y, theta));
    return theta, J_history;

  def featureNormalize(self, X):
    X_norm = X;
    mu = np.zeros(np.size(X, 2));
    sigma = np.zeros(np.size(X, 2));
    return sigma;

if __name__ == "__main__":
    linear_regression = week2Ex1();

    # data = pd.read_csv('ex1data1.txt',names=["X","y"])

    # X = data["X"];
    # y = data["y"];

    data = np.loadtxt('exercices/ex1data1.txt', delimiter=',');
    X, y = data[:, 0], data[:, 1];
    
    
    #data.plot(kind="scatter", x="X", y="y")

    #plt.show()

    m = y.size

    theta = np.zeros(2)
    X = np.stack([np.ones(m), X], axis=1)
    iterations = 1500;
    alpha = 0.01;
    
    J = linear_regression.computeCost(X,y,theta);

    print("with theta=[0 ; 0], Cost computed = %f", J)
    print('Expected cost value (approx) 32.07');

    #  further testing of the cost function
    J = linear_regression.computeCost(X,y,np.array([-1 , 2]));
    print("with theta=[-1 , 2], Cost computed = %f", J)

    theta, J_history = linear_regression.linearRegression(X, y, theta, alpha, iterations);

    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
    print('Expected theta values (approximately): [-3.6303, 1.1664]')

    linear_regression.plotData(X[:, 1], y)
    plt.plot(X[:, 1], np.dot(X, theta), '-')
    plt.legend(['Training data', 'Linear regression']);

    # plt.show()

    predict1 = [1, 3.5] *theta;
    print('For population = 35,000, we predict a profit of %f\n', predict1*10000);
    predict2 = [1, 7] * theta;
    print('For population = 70,000, we predict a profit of %f\n', predict2*10000);

    # grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i, theta0 in enumerate(theta0_vals):
      for j, theta1 in enumerate(theta1_vals):
        J_vals[i, j] = linear_regression.computeCost(X, y, [theta0, theta1])

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T
    
    # surface plot
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Surface')
    
    # contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax = plt.subplot(122)
    plt.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    plt.title('Contour, showing minimum')

    plt.show()
    pass
