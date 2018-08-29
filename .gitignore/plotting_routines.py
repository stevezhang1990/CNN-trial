#
# Introduction to Neural Networks.
# Given at SciNet, May 30 2017, by Erik Spence.
#
# This file, plotting_routines.py, contains some simple plotting
# routines.
#

#######################################################################


"""
plotting_routines.py contains two routines for plotting the class'
data.

"""


#######################################################################


import matplotlib.pyplot as plt
import numpy as np


#######################################################################


def plot_dots(x, v, **kwargs):

    """
    This function will generate a scatter plot of the data, with the
    colour of the dots indicating the category of the data point.

    Inputs:

    - x: 2D array of floats of shape (num_points, 2), containing the 2D
      position of the data points.  num_points is the number of data
      points.
    
    - v: integer vector of length num_points, containing the correct
      values (0 or 1) for the data.
    
    Outputs: nothing returned.

    """

    # Get the number of data points.
    num_points = len(x[:, 0])

    # Set min and max values and give it some padding.
    if 'x_min' in kwargs: x_min = kwargs['x_min']
    else: x_min = x[:, 0].min() * 1.1

    if 'x_max' in kwargs: x_max = kwargs['x_max']
    else: x_max = x[:, 0].max() * 1.1
    
    if 'y_min' in kwargs: y_min = kwargs['y_min']
    else: y_min = x[:, 1].min() * 1.1
   
    if 'y_max' in kwargs: y_max = kwargs['y_max']
    else: y_max = x[:, 1].max() * 1.1

    # Set the colours based on the v values.
    cy = np.array(['Orange'] * num_points)
    cy[v == 1] = 'Blue'

    # Plot the points, and tweak the axes.
    plt.scatter(x[:, 0], x[:, 1], c = cy, s = 50)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


#######################################################################


def plot_decision_boundary(x, v, model, predict_function, **kwargs):

    """
    This function generates a plot of the model's decision boundary,
    and then scatter plots the data on top of it.

    This function is heavily based on something I found on the web.
    Unfortunately, I can't remember where I found it.  Thanks to the
    author.

    Inputs:

    - x: 2D array of floats of shape (num_points, 2), containing the
      2D position of the data points.  num_points is the number of
      data points.
    
    - v: integer vector of length num_points, containing the correct
      values (0 or 1) for the data.
    
    - model: dictionary containing the model parameters.

    - predict_function: name of the function used to run the forward
      pass of the model.

    Outputs: nothing returned.

    """

    # Set min and max values and give it some padding
    if 'x_min' in kwargs: x_min = kwargs['x_min']
    else: x_min = x[:, 0].min() * 1.1

    if 'x_max' in kwargs: x_max = kwargs['x_max']
    else: x_max = x[:, 0].max() * 1.1
    
    if 'y_min' in kwargs: y_min = kwargs['y_min']
    else: y_min = x[:, 1].min() * 1.1
   
    if 'y_max' in kwargs: y_max = kwargs['y_max']
    else: y_max = x[:, 1].max() * 1.1

    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Join the x and y positions.
    c = np.c_[xx.ravel(), yy.ravel()]

    # Calculate the model values for the whole grid.  Round to the
    # nearest integer.
    yp = np.round(predict_function(c, model))
    yp = yp.reshape(xx.shape)

    # Plot the model contour and training data.
    plt.contourf(xx, yy, yp, cmap = plt.cm.Spectral)
    plot_dots(x, v)

    plt.show()
