from mimetypes import init
import numpy as np;
import utils
from matplotlib import pyplot as plt
import scipy.optimize as optimize

class Week4Ex1():
  
  def __init__(self):
     np.random.seed(1)


if __name__ == "__main__":
    one_vs_all = Week4Ex1();

    data = np.loadtxt('training_data/ex2data2.txt', delimiter=',');
    X, y = data[:, :2], data[:, 2]
    [m, n] = X.shape
 