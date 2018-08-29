#
# Introduction to Neural Networks.
# Given at SciNet, May 30 2017, by Erik Spence.
#
# This file, second_network.py, contains the implementation of our
# second neural network.
#

#######################################################################


"""
second_network contains the implementation of our
single-hidden-layer neural network.  Use 'build_model' to train the
network.

"""


#######################################################################


import numpy as np
import numpy.random as npr
    

#######################################################################


def sigma(z):

    """
    Returns the sigmoid function evaluated at z.

    Inputs:

    - z: vector of floats.

    Outputs:

    - vector of floats, the same length as z.

    """

    # Return the result.
    return 1. / (1. + np.exp(-z))


#######################################################################


def sigmaprime(z):

    """
    Returns the derivative of the sigmoid function, evaluated at z.

    Inputs:

    - z: vector of floats.

    Outputs:

    - vector of floats, the same length as z.

    """

    # Return the result.
    return sigma(z) * (1.0 - sigma(z))


#######################################################################


def forward(x, model):

    """This function runs a forward pass of the data through the neural
    network, and returns the values which were calculated along the
    way.

    Inputs:

    - x: 2D array of floats of shape (num_points, input_dims),
      containing the data to be input to the network.  num_points is
      the number of data points.  input_dims is the dimension of the
      input data.
    
    - model: dictionary containing the model parameters.  These model
      parameters should include:

        - 'w1': 2D array of floats of shape (num_nodes, input_dim).
          These are the weights for the hidden layer.

        - 'b1': 2D array of floats of shape (num_nodes, 1).  These are
          the biases for the hidden layer.  The superfluous extra
          dimension is needed so that the biases can be seamlessly
          added to the weights-data product.

        - 'w2': 2D array of floats of shape (output_dim, num_nodes).
          These are the weights for the output layer.

        - 'b2': 2D array of floats of shape (output_dim, 1).  These are
          the biases for the output layer.

    Outputs:

    - z1, z2, a1, a2, as a tuple.  These are 
    
        - z1: 2D array of floats of shape (num_nodes, num_points),
          containing the value of the variable z to be input to the
          hidden layer.  num_nodes is the number of nodes in the
          hidden layer.

        - z2: 2D array of floats of shape (output_dim, num_points),
          containing the value of the variable z to be input to the
          output layer.  output_dim is the output dimension of the
          network.

        - a1: 2D array of floats of shape (num_nodes, num_points),
          containing the output of the hidden layer.

        - a2: 2D array of floats of shape (output_dim, num_points),
          containing the output of the output layer.

    """

    # Forward propagation through the network.
    # First the hidden layer.
    z1 = model['w1'].dot(x.T) + model['b1']
    a1 = sigma(z1)

    # Then the output layer.
    z2 = model['w2'].dot(a1) + model['b2']
    a2 = sigma(z2)

    return z1, z2, a1, a2


#######################################################################


def predict(x, model):

    """
    The predict function runs the data through a forward pass of the
    neural network, and returns the output.  For our second network
    this means calculating the variable a2, and getting the maximum
    output values for each data point.

    Inputs:

    - x: 2D array of floats of shape (num_points, input_dims),
      containing the data to be input to the network.
    
    - model: dictionary containing the model parameters.

    Outputs:

    - vector of floats of length num_points.

    """

    # Run the data through the network, but we're only interested in
    # the output.
    _, _, _, a2 = forward(x, model)

    # Get the maximum value for each datapoint, and return it.
    return np.argmax(a2, axis = 0)


#######################################################################


# The first rule of Thesaurus Club is: do not discuss, confer about,
# descant, confabulate, converse about or mention Thesaurus Club.


#######################################################################


def build_model(num_nodes, x, v, eta, output_dim, num_steps = 10000,
                print_best = True, lam = 0.0):

    """
    This function uses gradient descent to update the neural network's
    model parameters, minimizing the quadradic cost function.  It
    returns the best model.

    Inputs:

    - num_nodes: integer, number of nodes in the hidden layer.

    - x: 2D array of floats of shape (num_points, input_dim),
      containing the input data.
    
    - v: integer vector of length num_points, containing the correct
      values (0 or 1) for the data.
    
    - eta: float, the stepsize parameter for the gradient descent.

    - output_dim: integer, number of nodes in the output layer.

    - num_steps: integer, number of steps to iterate through the
      training data for gradient descent.

    - print_best: boolean, if True, print the model accuracy every
      1000 iterations.

    - lam: float, regularization parameter.

    Outputs:

    - dictionary containing the parameters of the best model.

    """

    # Get the input dimension of the data.
    input_dim = np.shape(x)[1]
    
    # Initialize the parameters to random values. We need to learn
    # these.
    model = {'w1': npr.randn(num_nodes, input_dim),
             'b1': np.zeros([num_nodes, 1]),
             'w2': npr.randn(output_dim, num_nodes),
             'b2': np.zeros([output_dim, 1])}

    # A scaling factor used in determining the best model.
    scale = 100. / float(len(v))

    # Initialize the score of our best model.
    best = 0.0

    # Forward propagation.
    z1, _, a1, a2 = forward(x, model)
    
    # Gradient descent.
    for i in xrange(0, num_steps):

        # Backpropagation
        delta2 = a2
        # Here we subtract v, which is just 1, but only where v == 1.
        # This is the error in the final output (how wrong is it?).
        # (We should similarly subtract 0 where v == 0, but of course
        # this would not do anything.)
        delta2[v, range(len(v))] -= 1
        delta1 = (model['w2'].T).dot(delta2) * sigmaprime(z1)

        # Calculate the derivatives.
        dCdb2 = np.sum(delta2, axis = 1, keepdims = True)
        dCdb1 = np.sum(delta1, axis = 1, keepdims = True)
        
        dCdw2 = delta2.dot(a1.T)
        dCdw1 = delta1.dot(x)

        # Gradient descent parameter update, with regularization.
        model['w1'] -= eta * (lam * model['w1'] + dCdw1)
        model['b1'] -= eta * dCdb1
        model['w2'] -= eta * (lam * model['w2'] + dCdw2)
        model['b2'] -= eta * dCdb2

        # Check to see if this is our best model yet.
        z1, _, a1, a2 = forward(x, model)
        score = sum(np.argmax(a2, axis = 0) == v) * scale
        
        # Keep the best model.
        if (score > best):
            best, bestmodel = score, model.copy()

        # Optionally print the score.
        if (print_best) and (i % 1000 == 0):
            print "Best by step %i: %.1f %%" % (i, best)
     
    print "Our best model gets %.1f percent correct!" % best

    # Return the best parameters.
    return bestmodel

