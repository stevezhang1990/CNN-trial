#
# SciNet's DAT112, Neural Network Programming.
# Lecture 5, 1 May 2018.
# Erik Spence.
#
# This file, model7.py, contains a routine to generate the model for
# our first MNIST CNN Keras example.
#

#######################################################################


"""
model7.py contains a routine which generates the model for the class'
first MNIST CNN Keras example.

"""


#######################################################################


import keras.models as km
import keras.layers as kl


#######################################################################


def get_model(numfm, numnodes, input_shape = (28, 28, 1),
              output_size = 10):

    """
    This function returns a convolution neural network Keras model,
    with numfm feature maps and numnodes neurons in the
    fully-connected layer.

    Inputs:
    - numfm: int, the number of feature maps in the convolution layer.

    - numnodes: int, the number of nodes in the fully-connected layer.

    - intput_shape: tuple, the shape of the input data, 
    default = (28, 28, 1).

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()

    # Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size = (5, 5),
                        input_shape = input_shape,
                        activation = 'relu'))

    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (2, 2),
                              strides = (2, 2)))

    # Convert the network from 2D to 1D.
    model.add(kl.Flatten())

    # Add a fully-connected layer.
    model.add(kl.Dense(numnodes,
                       activation = 'tanh'))

    # Add the output layer.
    model.add(kl.Dense(10, activation = 'softmax'))

    # Return the model.
    return model


#######################################################################
