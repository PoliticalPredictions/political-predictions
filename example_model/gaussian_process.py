"""Linear regression model"""
import numpy as np


class GaussianProcessRegression(object):
    """Standard Gaussian Process Regression"""

    def __init__(self, kernel, *kernel_args, noise=1e-8):
        """Standard Gaussian Process Regression
        Implementation based on Algorithm 2.1 of Gaussian Processes
        for Machine Learning (GPML) by Rasmussen and Williams.

        Parameters
        ----------
        kernel : function(x, y, *kernel_args)
            Kernel used ot generate the Gram matrixes for GP regression
        *kernel_args
            Parameters for the kernel
        noise : float
            optional sigma^2 noise for of the measurements
        """
        self.noise = noise
        self.kernel = kernel
        self.kernel_args = kernel_args

    def fit(self, x_train, y_train):
        """Train the model on data

        Parameters
        ----------
        x_train : np.array
            Train data in dims (nexamples, nfeatures)
        y_train : np.array
            Labels in dims od (nexamples, 1)
        """

        # ------ Prepare covariance matrices
        npoints_train = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

        self.K = self.kernel(x_train, x_train, *self.kernel_args) + np.eye(npoints_train) * self.noise

        # ------ Compute the alpha
        self.L = np.linalg.cholesky(self.K)
        Ly = np.linalg.solve(self.L, y_train)
        self.alpha = np.linalg.solve(self.L.T, Ly)

        # # ------ Moe standard approach
        # self.Kinv = np.linalg.inv(self.K)
        # self.alpha2 = self.Kinv @ y_train

    def predict(self, x_predict, return_sample=False):
        """Make predictions using the model

        Parameters
        ----------
        x_predict : np.array
            Prediction points in dims (nexamples, nfeatures)
        return_sample : bool
            If True, returns a sample from multivariate Gaussian at predict points

        Returns
        -------
        np.array
            Mean at prediction points with dims od (nexamples, 1)
        np.array
            Sigmas at prediction points with dims od (nexamples, 1)
        np.array
            If return_sample is True, sample from a multivariate Gaussian with dims (nexamples, 1)

        """
        Ks = self.kernel(self.x_train, x_predict, *self.kernel_args)
        Kss = self.kernel(x_predict, x_predict, *self.kernel_args)

        self.mu = np.dot(Ks.T, self.alpha)

        v = np.linalg.solve(self.L, Ks)
        self.cov = Kss - np.dot(v.T, v)
        self.sigmas = np.sqrt(np.diag(self.cov)).reshape(-1, 1)

        # # ------ Second part of a more standard approach
        # self.mu2 = Ks.T @ self.alpha2
        # self.cov2 = Kss - Ks.T @ self.Kinv @ Ks
        # self.sigmas2 = np.sqrt(np.diag(self.cov2)).reshape(-1, 1)

        if return_sample:
            nsamples = self.cov.shape[0]
            covariance = self.cov + np.eye(nsamples) * 1e-10

            # In some version of Numpy, the multivariate normal is broken so use the scipy one instead
            # import scipy.stats
            # sample = scipy.stats.multivariate_normal.rvs(self.mu.flatten(), covariance).reshape(-1, 1)

            sample = np.random.multivariate_normal(self.mu.flatten(), covariance).reshape(-1, 1)
            return self.mu, self.sigmas, sample
        else:
            return self.mu, self.sigmas
