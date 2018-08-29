#
# Introduction to Neural Networks.
# Given at SciNet, May 30 2017, by Erik Spence.
#
# This file, mnist_loader.py, contains the code needed to load the
# MNIST dataset.  The code borrows heavily from
# http://neuralnetworksanddeeplearning.com.
#

#######################################################################


"""
mnist_loader contains the code needed to load the MNIST dataset,
both 1D and 2D versions.  The code has been heavily borrowed from
http://neuralnetworksanddeeplearning.com.

"""


#######################################################################


try:
    import cPickle
except:
    import pickle as cPickle

import gzip
import numpy as np


#######################################################################


def load_mnist_1D(filename = '../data/mnist.pkl.gz'):

    """
    Returns the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    Inputs:
    
    - filename: string, name of the file containing the data.

    Outputs:
    
    - tuple, containing the training, validation and test data.  These
      take the form:

        - training_data: tuple, consisting of:
   
            - 2D array of floats of shape (50000, 784), containing the
              pixel values for each image.  

            - integer vector of length 50000, containing the value of
              the number in the image.
       
        - validation_data: same as training_data, except with length
          10000

        - test_data: same as training_data, except with length
          10000

    """

    # Open the file.
    f = gzip.open(filename, 'rb')

    # Load the data.
    training_data, validation_data, test_data = cPickle.load(f)

    # Close the file.
    f.close()

    # Return the values.
    return training_data[0], training_data[1], \
        validation_data[0], validation_data[1], \
        test_data[0], test_data[1]


#######################################################################


def load_mnist_2D(filename = ''):

    """
    Returns the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    Inputs:
    
    - filename: string, name of the file containing the data.

    Outputs:
    
    - tuple, containing the training, validation and test data.  These
      take the form:

        - training_data: tuple, consisting of:
   
            - 2D array of floats of shape (50000, 48, 48, 1),
              containing the pixel values for each image.

            - integer vector of length 50000, containing the value of
              the number in the image.
       
        - validation_data: same as training_data, except with length
          10000

        - test_data: same as training_data, except with length
          10000

    """

    # Get the data.
    tr_d, tr_v, va_d, va_v, te_d, te_v = load_mnist_1D(filename = filename)

    # Reshape the data.
    training_inputs = np.array([x.reshape(28, 28, 1) for x in tr_d])
    validation_inputs = np.array([x.reshape(28, 28, 1) for x in va_d])
    test_inputs = np.array([x.reshape(28, 28, 1) for x in te_d])

    # Return the data.
    return training_inputs, tr_v, validation_inputs, va_v, \
        test_inputs, te_v

