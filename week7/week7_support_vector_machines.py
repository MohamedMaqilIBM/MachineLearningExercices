import os
print(os.getcwd())

import numpy as np
# Import regular expressions to process emails
import re

from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

class Week7():
  
  def __init__(self):
    np.random.seed(1)
  
  def gaussianKernel(self, x1, x2, sigma):
    sim = 0
    sim = np.exp(-np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2)))
    return sim

  def dataset3Params(self, X, y, Xval, yval):
    C = 1
    sigma = 0.3

    C_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma_array = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

    my_array = np.zeros([C_array.size, sigma_array.size])
    
    for i in np.arange(C_array.size):
        for j in np.arange(sigma_array.size):
            model= utils.svmTrain(X, y, C_array[i], self.gaussianKernel, args=(sigma_array[j],))
            predictions = utils.svmPredict(model, Xval)
            pred_error = np.mean(predictions != yval)
            
            my_array[i, j] = pred_error
        
    ind = np.unravel_index(np.argmin(my_array, axis = None), my_array.shape)
    C = C_array[ind[0]]
    sigma = sigma_array[ind[1]]

    return C, sigma

  def processEmail(self, email_contents, verbose=True):
    vocabList = utils.getVocabList()
    word_indices = []

    # Lower case
    email_contents = email_contents.lower()
    # Strip all HTML
    email_contents =re.compile('<[^<>]+>').sub(' ', email_contents)
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    # Handle URLS
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    # Handle $ sign
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    # get rid of any punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    # remove any empty word string
    email_contents = [word for word in email_contents if len(word) > 0]

    stemmer = utils.PorterStemmer()
    processed_email = []

    for word in email_contents:
      word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
      word = stemmer.stem(word)
      processed_email.append(word)

      if len(word) < 1:
        continue

      try:
          word_indices.append(vocabList.index(word))
      except ValueError:
          pass

    if verbose:
        print('----------------')
        print('Processed email:')
        print('----------------')
        print(' '.join(processed_email))

    return word_indices

  def emailFeatures(self, word_indices):
    n = 1899
    x = np.zeros(n)
    print(x)
    x[word_indices] = 1
    return x


if __name__ == "__main__":
    support_vector_machines = Week7();

    # Load from ex6data1
    # You will have X, y as keys in the dict data
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y'][:, 0]

    # # Plot training data
    # utils.plotData(X, y)

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1

    model = utils.svmTrain(X, y, C, utils.linearKernel, 1e-3, 20)
    # utils.visualizeBoundaryLinear(X, y, model)

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2

    sim = support_vector_machines.gaussianKernel(x1, x2, sigma)

    print('Gaussian Kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = %0.2f:'
          '\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

    # Load from ex6data2
    # You will have X, y as keys in the dict data
    data = loadmat('ex6data2.mat')
    X, y = data['X'], data['y'][:, 0]

    # # Plot training data
    # utils.plotData(X, y)

    # SVM Parameters
    C = 1
    sigma = 0.1

    model= utils.svmTrain(X, y, C, support_vector_machines.gaussianKernel, args=(sigma,))
    # utils.visualizeBoundary(X, y, model)

    # Load from ex6data3
    # You will have X, y, Xval, yval as keys in the dict data
    data = loadmat('ex6data3.mat')
    X, y, Xval, yval = data['X'], data['y'][:, 0], data['Xval'], data['yval'][:, 0]

    # # Plot training data
    # utils.plotData(X, y)

    # Try different SVM Parameters here
    C, sigma = support_vector_machines.dataset3Params(X, y, Xval, yval)

    # Train the SVM
    # model = utils.svmTrain(X, y, C, lambda x1, x2: gaussianKernel(x1, x2, sigma))
    model = utils.svmTrain(X, y, C, support_vector_machines.gaussianKernel, args=(sigma,))
    utils.visualizeBoundary(X, y, model)
    print(C, sigma)

    # Extract Features
    with open(os.path.join('training_data', 'emailSample1.txt')) as fid:
        file_contents = fid.read()

    word_indices  = support_vector_machines.processEmail(file_contents)

    #Print Stats
    print('-------------')
    print('Word Indices:')
    print('-------------')
    print(word_indices)

    # Extract Features
    with open(os.path.join('training_data', 'emailSample1.txt')) as fid:
        file_contents = fid.read()

    word_indices  = support_vector_machines.processEmail(file_contents)
    features      = support_vector_machines.emailFeatures(word_indices)

    # Print Stats
    print('\nLength of feature vector: %d' % len(features))
    print('Number of non-zero entries: %d' % sum(features > 0))

    # Load the Spam Email dataset
    # You will have X, y in your environment
    data = loadmat(os.path.join('training_data', 'spamTrain.mat'))
    X, y= data['X'].astype(float), data['y'][:, 0]

    print('Training Linear SVM (Spam Classification)')
    print('This may take 1 to 2 minutes ...\n')

    C = 0.1
    model = utils.svmTrain(X, y, C, utils.linearKernel)

    # Compute the training accuracy
    p = utils.svmPredict(model, X)

    print('Training Accuracy: %.2f' % (np.mean(p == y) * 100))

    # Load the test dataset
    # You will have Xtest, ytest in your environment
    data = loadmat(os.path.join('Data', 'spamTest.mat'))
    Xtest, ytest = data['Xtest'].astype(float), data['ytest'][:, 0]

    print('Evaluating the trained Linear SVM on a test set ...')
    p = utils.svmPredict(model, Xtest)

    print('Test Accuracy: %.2f' % (np.mean(p == ytest) * 100))

    # Sort the weights and obtin the vocabulary list
    # NOTE some words have the same weights, 
    # so their order might be different than in the text above
    idx = np.argsort(model['w'])
    top_idx = idx[-15:][::-1]
    vocabList = utils.getVocabList()
    
    print('Top predictors of spam:')
    print('%-15s %-15s' % ('word', 'weight'))
    print('----' + ' '*12 + '------')
    for word, w in zip(np.array(vocabList)[top_idx], model['w'][top_idx]):
        print('%-15s %0.2f' % (word, w))




