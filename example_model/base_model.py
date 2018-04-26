"""BAse class for ML models"""


class BaseModel(object):
    """Base class for Models"""

    def fit(self, x, y):
        """Train the model on data

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)
        y : np.array
            Labels in dims od (nexamples, 1)
        """
        pass

    def predict(self, x):
        """Make predictions using the model

        Parameters
        ----------
        x : np.array
            input data in dims (nexamples, nfeatures)

        Returns
        -------
        np.array
            predictions in dims od (nexamples, 1)
        """
        return self._forward(x)

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
        pass
