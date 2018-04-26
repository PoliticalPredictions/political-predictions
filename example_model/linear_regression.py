"""Linear regression model"""
import numpy as np
from .base_model import BaseModel


class LinearRegression(BaseModel):
    """Linear Regression model"""

    def __init__(self, nfeatures, regularisation=None):
        """Summary

        Parameters
        ----------
        nfeatures : int
            Number of features in the examples
        regularisation : mlp.Regularisation
            regularisation class
        """
        self.nfeatures = nfeatures
        self.w = np.random.random((nfeatures, 1)) * 0.01
        self.b = 0
        if regularisation is None:
            self.regu = NoRegularisation(0)
        else:
            self.regu = regularisation

    def _forward(self, x):
        """Propagate forward

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)

        Returns
        -------
        np.array
            predictions in dims od (nexamples, 1)
        """
        return np.dot(x, self.w) + self.b

    def cost(self, a, y):
        """Compute the cost

        Parameters
        ----------
        a : np.array
            predictions in dims od (nexamples, 1)
        y : np.array
            Labels in dims od (nexamples, 1)

        Returns
        -------
        float
            cost
        """
        nexamples = y.shape[0]
        return (0.5 / nexamples) * np.sum(np.square(a - y)) + self.regu.cost(self.w)

    def backward(self, x, forward_result, y):
        """Compute the gradient on weigths

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)
        forward_result : np.array
            predictions in dims od (nexamples, 1)
        y : np.array
            Labels in dims od (nexamples, 1)

        Returns
        -------
        np.array
            Gradient in dims of (nfeatures, 1)
        """
        nexamples = y.shape[0]
        cost_g = (1 / nexamples) * (forward_result - y)
        w_grad = np.dot(x.T, cost_g) + self.regu.gradient(self.w)
        b_grad = np.sum(cost_g)
        return w_grad, b_grad

    def full_pass(self, x, y):
        """Calculate the cost and gradient

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)
        y : np.array
            Labels in dims od (nexamples, 1)

        Returns
        -------
        float
            cost
        np.array
            Gradient in dims of (nfeatures, 1)
        """
        forward = self._forward(x)
        cost = self.cost(forward, y)
        w_grad, b_grad = self.backward(x, forward, y)

        return cost, w_grad, b_grad

    def fit(self, x, y, epochs=100, learning_rate=1e-2, print_every=100):
        """fit the model on data

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)
        y : np.array
            Labels in dims od (nexamples, 1)
        epochs : int
            Number of epochs
        learning_rate : float
            Learning rate for the SGD
        print_every : int
            print information every N epochs
        """
        cost = 0
        for idx in range(int(epochs)):
            c, wg, bg = self.full_pass(x, y)
            cost += c
            self.w -= wg * learning_rate
            self.b -= bg * learning_rate
            if ((idx + 1) % print_every == 0):
                print("Cost after {} epochs: {:.3e}".format(idx + 1, cost / print_every))
                cost = 0


class NoRegularisation(object):
    """No added regularisation"""

    def __init__(self, lbd):
        self.lbd = lbd

    def cost(self, w):
        """Calculate cost addition due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """

        reg = 0
        return reg

    def gradient(self, w):
        """Calculate weigths gradient due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """
        reg = 0
        return reg


class RidgeRegularisation(NoRegularisation):
    """Gaussian prior resultng in L2 regularisation"""

    def cost(self, w):
        """Calculate cost addition due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """

        reg = np.sum(np.square(w)) * self.lbd / 2
        return reg

    def gradient(self, w):
        """Calculate weigths gradient due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """
        reg = self.lbd * np.abs(w)
        return reg


class LassoRegularisation(NoRegularisation):
    """Laplace prior resultng in L1 regularisation"""

    def cost(self, w):
        """Calculate cost addition due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """

        reg = np.sum(np.abs(w)) * self.lbd
        return reg

    def gradient(self, w):
        """Calculate weigths gradient due to reguarisation

        Parameters
        ----------
        w : np.array
            Linear regression weigths

        Returns
        -------
        np.arrar
            increase in cost
        """
        reg = self.lbd * np.sign(w)
        return reg
