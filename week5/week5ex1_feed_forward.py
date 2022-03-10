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
  
  def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.0):
    """"
    Instructions
    ------------
    You should complete the code by working through the following parts.
    
    - Part 1: Feedforward the neural network and return the cost in the 
              variable J. After implementing Part 1, you can verify that your
              cost function computation is correct by verifying the cost
              computed in the following cell.
    
    - Part 2: Implement the backpropagation algorithm to compute the gradients
              Theta1_grad and Theta2_grad. You should return the partial derivatives of
              the cost function with respect to Theta1 and Theta2 in Theta1_grad and
              Theta2_grad, respectively. After implementing Part 2, you can check
              that your implementation is correct by running checkNNGradients provided
              in the utils.py module.
    
              Note: The vector y passed into the function is a vector of labels
                    containing values from 0..K-1. You need to map this vector into a 
                    binary vector of 1's and 0's to be used with the neural network
                    cost function.
     
              Hint: We recommend implementing backpropagation using a for-loop
                    over the training examples if you are implementing it for the 
                    first time.
    
    - Part 3: Implement regularization with the cost function and gradients.
    
              Hint: You can implement this around the code for
                    backpropagation. That is, you can compute the gradients for
                    the regularization separately and then add them to Theta1_grad
                    and Theta2_grad from Part 2.
    """
    J = 0
    return J

if __name__ == "__main__":
    feed_forward = Week5Ex1();

    #  training data stored in arrays X, y
    data = loadmat(os.path.join('training_data', 'ex4data1.mat'))
    X, y = data['X'], data['y'].ravel()

    # set the zero digit to 0, rather than its mapped 10 in this dataset
    # This is an artifact due to the fact that this dataset was used in 
    # MATLAB where there is no index 0
    y[y == 10] = 0

    # Number of training examples
    m = y.size

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

    a = np.array([1,2,3])
    b = ([3,4,5])
    print(a*b)