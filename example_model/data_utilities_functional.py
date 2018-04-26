"""Utilities for the example model"""
import numpy as np


def generate_design_matrix(n_training, n_cv, n_testing,
                           function, mini=0, maxi=1, max_noise=0, ndims=1):
    """Generate a desing matrix given the dataset

    Parameters
    ----------
    n_training : int
        Number of training examples
    n_cv : int
        Number of Cross Validation examples
    n_testing : int
        Number of Test examples
    function : function
        Any numpy function
    mini : float
        datapoints will be drawn from (mini - maxi) range in each dim
    maxi : float
        datapoints will be drawn from (mini - maxi) range in each dim
    max_noise : float
        noise of random() * noise will be added
    ndims : int
        number of dimensions for the function

    Returns
    -------
    x_train : np.array
        Train data in dims (nexamples, nfeatures)
    y_train : np.array
        Labels in dims od (nexamples, 1)
    x_cv : np.array
        Cross Validation data in dims (nexamples, nfeatures)
    y_cv : np.array
        Labels in dims od (nexamples, 1)
    x_test : np.array
        test data in dims (nexamples, nfeatures)
    y_test : np.array
        Labels in dims od (nexamples, 1)
    """

    x_train, y_train = generate_data(function, mini, maxi, max_noise, ndims, n_training)
    x_cv, y_cv = generate_data(function, mini, maxi, max_noise, ndims, n_cv)
    x_test, y_test = generate_data(function, mini, maxi, max_noise, ndims, n_testing)

    return x_train, y_train, x_cv, y_cv, x_test, y_test


def generate_data(function, mini, maxi, max_noise, ndims, npoints):
    """Generate data by adding noise to a given single values function

    Parameters
    ----------
    function : function
        Any numpy function
    mini : float
        datapoints will be drawn from (mini - maxi) range in each dim
    maxi : float
        datapoints will be drawn from (mini - maxi) range in each dim
    max_noise : float
        noise of random() * noise will be added
    ndims : int
        number of dimensions for the function
    npoints : int
        number of data points

    Returns
    -------
    X: np.array
        input points given as (npoints, ndimensions)
    Y: np.array
        function values given in (npoints, 1)
    """
    X = np.random.rand(npoints, ndims) * (maxi - mini) + mini
    Y = np.zeros((npoints, 1))
    for idx in range(npoints):
        ns = np.random.random_sample() * max_noise
        Y[idx, 0] = function(X[idx, :]) + ns

    return X, Y
