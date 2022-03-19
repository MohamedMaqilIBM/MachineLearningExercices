# used for manipulating directory paths
import os

import numpy as np
import utils
from matplotlib import pyplot as plt
import scipy.optimize as optimize
# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

class Week5Ex1():
  
  def __init__(self):
    np.random.seed(1)

  def sigmoidGradient(self, z):
    g = np.zeros(z.shape)
    g = utils.sigmoid(z) * (1 - utils.sigmoid(z))
    return g

  def nnCostFunction(self, nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.0):

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size
         
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = utils.sigmoid(a1.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    a3 = utils.sigmoid(a2.dot(Theta2.T))
    
    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]
    
    temp1 = Theta1
    temp2 = Theta2
    
    # Add regularization term
    reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))
    
    J = (-1 / m) * np.sum((np.log(a3) * y_matrix) + np.log(1 - a3) * (1 - y_matrix)) + reg_term
    
    # Backpropogation
    delta_3 = a3 - y_matrix
    delta_2 = delta_3.dot(Theta2)[:, 1:] * self.sigmoidGradient(a1.dot(Theta1.T))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)
    
    # Add regularization to gradient
    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]

    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]

    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
    return J, grad
  


  def randInitializeWeights(self, L_in, L_out, epsilon_init=0.12):
    L_in_param = 1 + L_in
    W = np.zeros((L_out, L_in_param))
    W = np.random.rand(L_out, L_in_param) * 2 * epsilon_init - epsilon_init
    return W

if __name__ == "__main__":
    feed_forward = Week5Ex1();

    data = loadmat(os.path.join('training_data', 'ex4data1.mat'))
    X, y = data['X'], data['y'].ravel()

    y[y == 10] = 0
    m = y.size

    # Randomly select 100 data points to display
    rand_indices = np.random.choice(m, 100, replace=False)
    sel = X[rand_indices, :]
    utils.displayData(sel)

    # Setup the parameters you will use for this exercise
    input_layer_size  = 400  # 20x20 Input Images of Digits
    hidden_layer_size = 25   # 25 hidden units
    num_labels = 10          # 10 labels, from 0 to 9

    # Load the weights into variables Theta1 and Theta2
    weights = loadmat(os.path.join('training_data', 'ex4weights.mat'))

    # Theta1 has size 25 x 401
    # Theta2 has size 10 x 26
    Theta1, Theta2 = weights['Theta1'], weights['Theta2']

    # swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
    # since the weight file ex3weights.mat was saved based on MATLAB indexing
    Theta2 = np.roll(Theta2, 1, axis=0)

    # Unroll parameters 
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    lambda_ = 0
    J, _ = feed_forward.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
    print('Cost at parameters (loaded from ex4weights): %.6f ' % J)
    print('The cost should be about                   : 0.287629.')

    # Weight regularization parameter (we set this to 1 here).
    lambda_ = 1
    J, _ = feed_forward.nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                      num_labels, X, y, lambda_)

    print('Cost at parameters (loaded from ex4weights): %.6f' % J)
    print('This value should be about                 : 0.383770.')

    print('Initializing Neural Network Parameters ...')
    initial_Theta1 = feed_forward.randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = feed_forward.randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

    utils.checkNNGradients(feed_forward.nnCostFunction)

    #  Check gradients by running checkNNGradients
    lambda_ = 3
    utils.checkNNGradients(feed_forward.nnCostFunction, lambda_)

    # Also output the costFunction debugging values
    debug_J, _  = feed_forward.nnCostFunction(nn_params, input_layer_size,
                              hidden_layer_size, num_labels, X, y, lambda_)

    print('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' % (lambda_, debug_J))
    print('(for lambda = 3, this value should be about 0.576051)')

    options= {'maxiter': 100}
    lambda_ = 1

    costFunction = lambda p: feed_forward.nnCostFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            num_labels, X, y, lambda_)
    
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)
    
    # get the solution of the optimization
    nn_params = res.x
            
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    
    pred = utils.predict(Theta1, Theta2, X)
    print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

    utils.displayData(Theta1[:, 1:])

