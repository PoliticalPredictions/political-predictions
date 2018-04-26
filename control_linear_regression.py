"""Control script for using the example model"""
import example_model as model
import numpy as np


def main():
    mini = -2
    maxi = 2
    max_noise = 1e-6
    n_features = 1

    n_training = 100
    n_cv = 20
    n_testing = 20

    def function(x): return 2 * x + 1

    x_train, y_train, x_cv, y_cv, x_test, y_test = model.generate_design_matrix(n_training,
                                                                                n_cv,
                                                                                n_testing,
                                                                                function,
                                                                                mini=mini,
                                                                                maxi=maxi,
                                                                                max_noise=max_noise,
                                                                                ndims=n_features)

    lin_reg = model.LinearRegression(x_train.shape[1], regularisation=model.RidgeRegularisation(1e-3))
    lin_reg.fit(x_train, y_train,
                epochs=1000,
                learning_rate=1e-2,
                print_every=100)

    prediction = lin_reg.predict(x_test)
    error = np.sqrt(np.sum(np.square(prediction - y_test)))
    print("The error is {}".format(error))
    print("The weigths are y = Ax + B")
    print("A:", lin_reg.w)
    print("B:", lin_reg.b)


if (__name__ == "__main__"):
    main()
