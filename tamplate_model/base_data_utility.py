"""Data Utilities for the tamplate model"""
import numpy as np


def generate_design_matrix(dataset, **kwargs):
    """Generate a desing matrix given the dataset

    Parameters
    ----------
    dataset : any
        Raw data from the dataset
    **kwargs
        keyword arguments for data processing

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

    Raises
    ------
    NotImplementedError
        This is a tamplate
    """
    raise NotImplementedError("This is a tamplate")

    x_train = None
    y_train = None
    x_cv = None
    y_cv = None
    x_test = None
    y_test = None

    return x_train, y_train, x_cv, y_cv, x_test, y_test
