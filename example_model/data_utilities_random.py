"""Data Utilities for the example model"""
import numpy as np


def generate_uniform_matrix(rows, cols, mini=0, maxi=1):
    """Generate a (rows, cols) matrix with uniform entries

    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns
    mini : int
        Minimum value of random variable
    maxi : int
        Maximum value of random variable

    Returns
    -------
    x : np.array
        Random rray with dims (rows, cols)
    """
    x = (mini - maxi) * np.random.random_sample((rows, cols)) + mini
    return x


def generate_uniform_dataset(n_training, n_cv, n_testing, ndims=10, mini=0, maxi=1):
    """Generate a random dataset with random labels

    Parameters
    ----------
    n_training : int
        Number of training examples
    n_cv : int
        Number of Cross Validation examples
    n_testing : int
        Number of Test examples
    ndims : int
        Number of features in the dtaset
    mini : int
        Minimum value of random variable
    maxi : int
        Maximum value of random variable

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

    x_train = generate_uniform_matrix(n_training, ndims, mini, maxi)
    y_train = generate_uniform_matrix(n_training, 1, mini, maxi)

    x_cv = generate_uniform_matrix(n_cv, ndims, mini, maxi)
    y_cv = generate_uniform_matrix(n_cv, 1, mini, maxi)

    x_test = generate_uniform_matrix(n_testing, ndims, mini, maxi)
    y_test = generate_uniform_matrix(n_testing, 1, mini, maxi)

    return x_train, y_train, x_cv, y_cv, x_test, y_test
