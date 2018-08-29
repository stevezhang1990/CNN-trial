#
# Introduction to Neural Networks.
# Given at SciNet, May 30 2017, by Erik Spence.
#
# This file, first_network.py, contains the implementation of our
# first neural network.
#

#######################################################################


"""
first_network contains the implementation of our single-node neural
network.  Use 'build_model' to train the network.
"""


#######################################################################


import numpy as np
import numpy.random as npr


#######################################################################


def sigma(x, model):

    """
    Returns the sigmoid function evaluated at z, where z is the
    product of the model parameters with x.

    Inputs:

    - x: 2D array of floats of shape (num_points, 2), containing the
      2D position of the data points.  num_points is the number of
      data points.
    
    - model: dictionary containing the model parameters.  These model
      parameters should include:

        - 'w1': float, weight which multiplies the x dimension of the
          data.

        - 'w2': float, weight which multiplies the y dimension of the
          data.

        - 'b1': float, bias for the network.

    Outputs:

    - vector of floats of length num_points.

    """

    # Calculate z.
    z = model['w1'] * x[:,0] + model['w2'] * x[:,1] + model['b']

    # Return the result.
    return 1. / (1. + np.exp(-z))


#######################################################################


# The prediction function.  This function runs the data, in the
# forward direction, through the neural network.  Though it is
# obviously redundant for our first neural network, it is included for
# consistency with the second neural network example.
def predict(x, model):

    """
    The predict function runs the data through a forward pass of the
    neural network, and returns the output.  For our first network
    this simply means invoking the sigmoid function on the input data.

    Inputs:

    - x: 2D array of floats of shape (num_points, 2), containing the
      2D position of the data points.  num_points is the number of
      data points.
    
    - model: dictionary containing the model parameters.

    Outputs:

    - vector of floats of length num_points.

    """

    # Return the sigma function.
    return sigma(x, model)


#######################################################################


# Chomsky, Heisenberg and Goedel walk into a bar.
#
# Heisenberg says: "I can tell we're in a joke, but I can't tell if
# it's funny."
#
# Goedel says: "We can't tell if it's funny because we're inside the
# joke."
#
# Chomsky says: "The joke's funny, you're just not telling it right."


#######################################################################


def build_model(x, v, eta = 0.01, num_steps = 10000,
                print_best = True):

    """
    This function uses gradient descent to update the neural network's
    model parameters, minimizing the quadradic cost function.  It
    returns the best model.

    Inputs:

    - x: 2D array of floats of shape (num_points, 2), containing the
      2D position of the data points.  num_points is the number of
      data points.
    
    - v: integer vector of length num_points, containing the correct
      values (0 or 1) for the data.
    
    - eta: float, the stepsize parameter for the gradient descent.

    - num_steps: integer, number of steps to iterate through the
      training data for gradient descent.

    - print_best: boolean, if True, print the model accuracy every
      1000 iterations.

    Outputs:

    - dictionary containing the parameters of the best model.

    """
     
    # Initialize the parameters to random values. We need to learn
    # these.
    model = {'w1': npr.random(), 'w2': npr.random(),
             'b': npr.random()}

    # A scaling factor used in determining the best model.
    scale = 100. / float(len(v))

    # Initialize the score of our best model.
    best = 0.0

    # Forward propagation, to initialize f.
    f = sigma(x, model)
     
    # Gradient descent.
    for i in xrange(0, num_steps):

        # Calculate the derivatives.
        temp = (f - v) * f * (1 - f)
        dCdw1 = sum(temp * x[:, 0])
        dCdw2 = sum(temp * x[:, 1])
        dCdb  = sum(temp)

        # Update the parameters
        model['w1'] -= eta * dCdw1
        model['w2'] -= eta * dCdw2
        model['b']  -= eta * dCdb

        # Check to see if this is our best model yet.
        f = sigma(x, model)
        score = sum(np.round(f) == v) * scale
        
        # Keep the best model.
        if (score > best):
            best, bestmodel = score, model.copy()

        # Optionally print the score.
        if (print_best) and (i % 1000 == 0):
            print "Best by step %i: %.1f %%" % (i, best)
     
    print "Our best model gets %.1f percent correct!" % best

    # Return the best parameters
    return bestmodel

