"""Contro lscript for initial stages of development"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pickle
import kaggle_toolkit as kt


def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.square(np.log(y_pred + 1) - np.log(y_true + 1)).mean() ** 0.5


def load_dataset(name, split=True):
    """Load the pickled dataset and use a subset of it

    Returns
    -------
    If split is True:
    x_train : np.array
        Train data in dims (nexamples, nfeatures)
    y_train : np.array
        Labels in dims od (nexamples, 1)
    x_test : np.array
        test data in dims (nexamples, nfeatures)
    y_test : np.array
        Labels in dims od (nexamples, 1)

    If split is False:
    train_dataset : kaggle_toolkit.DescriptorDataset
        train data for the problem
    test_dataset : kaggle_toolkit.DescriptorDataset
        test data for the problem

    Parameters
    ----------
    name : str
        name of the dataset as "dataset/{name}_train.pkl"
    split : bool
        if True, returns
            x_train, y_train, x_test, y_test
        otherwise, returns
            train_dataset, test_dataset

    """
    with open("dataset/{}_train.pkl".format(name), "rb") as fp:
        train_dataset = pickle.load(fp, encoding='latin1')
    with open("dataset/{}_test.pkl".format(name), "rb") as fp:
        test_dataset = pickle.load(fp, encoding='latin1')

    if split:
        x_train = train_dataset.descriptors
        y_train = train_dataset.properties

        x_test = test_dataset.descriptors
        y_test = test_dataset.properties

        return x_train, y_train, x_test, y_test
    else:
        return train_dataset, test_dataset


def print_results(mu, y_test, name="Model"):
    """Print the results of predictions

    Parameters
    ----------
    mu : np.array
        Predictions in shape (nExamples, 1)
    y_test : np.array
        True Labels in shape (nExamples, 1)
    name : str
        Identification of the test
    """
    print(name + ": RMSLE error: {:.4e}".format(rmsle(mu, y_test)))
    print(name + ": Values(m,std)            :({:.4e}, {:.4e})".format(
        np.mean(np.abs(y_test)),
        np.std(y_test)))
    print(name + ": Rel RMSLE (1/mean, 1/std):({:.4e}, {:.4e})".format(
        rmsle(mu, y_test) / np.mean(np.abs(y_test)),
        rmsle(mu, y_test) / np.std(y_test)))


def single_test(name, noise, degree):
    """Run a test with a single set of parameters and print the results

    Parameters
    ----------
    name : str
        Name of the test
    noise : float
        Noise/regularisation parameter value
    degree : int
        Polynomial kernel degree value
    """
    print("Fitting for {}".format(name))
    x_train, y_train, x_test, y_test = load_dataset(name)

    kernel = kt.poly_kernel
    kernel_params = [degree]

    print("Fitting GP")
    gp_model = kt.GaussianProcessRegression(kernel, kernel_params, noise=noise)
    gp_model.fit(x_train, y_train)

    mu, sigmas = gp_model.predict(x_train, return_sample=False)
    print_results(mu, y_train, name="Training set")

    mu, sigmas = gp_model.predict(x_test, return_sample=False)
    print_results(mu, y_test, name="Test set    ")


def optimize_hyperparameters(x_train, y_train, x_test, y_test, valuesA, valuesB, verbose=False):
    """Perform  grid search for kernel paremeters

    Parameters
    ----------
    x_train : np.array
        Train data in dims (nexamples, nfeatures)
    y_train : np.array
        Labels in dims od (nexamples, 1)
    x_test : np.array
        test data in dims (nexamples, nfeatures)
    y_test : np.array
        Labels in dims od (nexamples, 1)
    valuesA : list of floats
        Noise/regularisation parameter values
    valuesB : list of ints
        Polynomial kernel degree values
    verbose : bool
        Turns on verbose mode

    Returns
    -------
    best_score : dictionary
        Dictionary with values of:
            best_score["value"]: lowest error
            best_score["indexa"]: regularisation param for the lowest error
            best_score["indexb"]: degree param for the lowest error
    results : np.array
        Individual values in dims od (nparamsA * nparamsB, 3)
            results[0, :]: indexA
            results[1, :]: indexB
            results[2, :]: score
    """
    best_score = dict(value=1e20,
                      indexa=0,
                      indexb=0)

    results = np.zeros((len(valuesA) * len(valuesB), 3))
    idx = 0

    for indexB in valuesB:
        for indexA in valuesA:
            if verbose:
                print("Testing GP for {}, {}".format(indexA, indexB))
            noise = indexA
            kernel = kt.poly_kernel
            kernel_params = [indexB]

            gp_model = kt.GaussianProcessRegression(kernel, kernel_params, noise=noise)
            gp_model.fit(x_train, y_train)

            mu, sigmas = gp_model.predict(x_test, return_sample=False)
            score = rmsle(mu, y_test)

            results[idx, 0] = indexA
            results[idx, 1] = indexB
            results[idx, 2] = score
            idx += 1

            if score < best_score["value"]:
                best_score["value"] = score
                best_score["indexa"] = indexA
                best_score["indexb"] = indexB

    return best_score, results


def cross_validate_hyperparameters(dataset, valuesA, valuesB,
                                   n_training, n_testing, n_iter,
                                   verbose=False):
    """Split a given dataset randomly and perform cross validation in search
    for parameters

    Parameters
    ----------
    dataset : kaggle_toolkit.DescriptorDataset
        train data for the problem
    valuesA : list of floats
        Noise/regularisation parameter values
    valuesB : list of ints
        Polynomial kernel degree values
    n_training : int
        number of training examples to extract from the set
    n_testing : int
        number of test examples to extract from the set
    n_iter : int
        number of repeats needed for mean standard deviation of results
    verbose : bool
        Turns on verbose mode

    Returns
    -------
    best_score : dictionary
        Dictionary with values of:
            best_score["value"]: lowest error
            best_score["indexa"]: regularisation param for the lowest error
            best_score["indexb"]: degree param for the lowest error
    results : np.array
        Individual values in dims od (nparamsA * nparamsB, 4)
            results[0, :]: indexA
            results[1, :]: indexB
            results[2, :]: scores mean
            results[2, :]: scores standard deviation
    """
    best_score = dict(value=1e20,
                      indexa=0,
                      indexb=0)

    results = np.zeros((len(valuesA) * len(valuesB), 4))
    master_idx = 0

    for indexB in valuesB:
        for indexA in valuesA:

            if verbose:
                print("Testing GP for {}, {}".format(indexA, indexB))
            noise = indexA
            kernel = kt.poly_kernel
            kernel_params = [indexB]

            scores = []
            for idx in range(n_iter):
                train_dataset, test_dataset = kt.split_dataset(dataset, n_training, n_testing)

                x_train = train_dataset.descriptors
                y_train = train_dataset.properties

                x_test = test_dataset.descriptors
                y_test = test_dataset.properties

                gp_model = kt.GaussianProcessRegression(kernel, kernel_params, noise=noise)
                gp_model.fit(x_train, y_train)

                mu, sigmas = gp_model.predict(x_test, return_sample=False)
                score = rmsle(mu, y_test)
                scores.append(score)

            score = np.mean(scores)
            stdev = np.std(scores)
            results[master_idx, 0] = indexA
            results[master_idx, 1] = indexB
            results[master_idx, 2] = score
            results[master_idx, 3] = stdev
            master_idx += 1

            if score < best_score["value"]:
                best_score["value"] = score
                best_score["indexa"] = indexA
                best_score["indexb"] = indexB

    return best_score, results


def test_set_cross_validate_hyperparameters(train_dataset, test_dataset, valuesA, valuesB,
                                            n_testing, n_iter,
                                            verbose=False):
    """Split a test dataset randomly and perform find the right parameters for the split

    Parameters
    ----------
    train_dataset : kaggle_toolkit.DescriptorDataset
        train data for the problem
    test_dataset : kaggle_toolkit.DescriptorDataset
        test data for the problem
    valuesA : list of floats
        Noise/regularisation parameter values
    valuesB : list of ints
        Polynomial kernel degree values
    n_testing : int
        number of testing examples to extract from the test set
    n_iter : int
        number of repeats needed for mean standard deviation of results
    verbose : bool
        Turns on verbose mode

    Returns
    -------
    best_score : dictionary
        Dictionary with values of:
            best_score["value"]: lowest error
            best_score["indexa"]: regularisation param for the lowest error
            best_score["indexb"]: degree param for the lowest error
    results : np.array
        Individual values in dims od (nparamsA * nparamsB, 4)
            results[0, :]: indexA
            results[1, :]: indexB
            results[2, :]: scores mean
            results[2, :]: scores standard deviation
    """
    best_score = dict(value=1e20,
                      indexa=0,
                      indexb=0)

    results = np.zeros((len(valuesA) * len(valuesB), 4))
    master_idx = 0

    x_train = train_dataset.descriptors
    y_train = train_dataset.properties

    for indexB in valuesB:
        for indexA in valuesA:

            if verbose:
                print("Testing GP for {}, {}".format(indexA, indexB))
            noise = indexA
            kernel = kt.poly_kernel
            kernel_params = [indexB]

            scores = []
            for idx in range(n_iter):
                subset_test_dataset, _ = kt.split_dataset(test_dataset, n_testing, 0)

                x_test = subset_test_dataset.descriptors
                y_test = subset_test_dataset.properties

                gp_model = kt.GaussianProcessRegression(kernel, kernel_params, noise=noise)
                gp_model.fit(x_train, y_train)

                mu, sigmas = gp_model.predict(x_test, return_sample=False)
                score = rmsle(mu, y_test)
                scores.append(score)

            score = np.mean(scores)
            stdev = np.std(scores)
            results[master_idx, 0] = indexA
            results[master_idx, 1] = indexB
            results[master_idx, 2] = score
            results[master_idx, 3] = stdev
            master_idx += 1

            if score < best_score["value"]:
                best_score["value"] = score
                best_score["indexa"] = indexA
                best_score["indexb"] = indexB

    return best_score, results


def plot_hyperparameters_search_results(name, valuesA, valuesB,
                                        res, bs, res_test, bs_test, use_test_set):
    """Plot the results of the hyperparameter search study

    Parameters
    ----------
    name : str
        Name of datasets to be used
    valuesA : list of floats
        Noise/regularisation parameter values
    valuesB : list of ints
        Polynomial kernel degree values
    res : np.array
        Individual results for the CV computations in dims od (nparamsA * nparamsB, 4)
            results[0, :]: indexA
            results[1, :]: indexB
            results[2, :]: scores mean
            results[2, :]: scores standard deviation
    bs : dictionary
        Dictionary with values of best results for the CV study:
            best_score["value"]: lowest error
            best_score["indexa"]: regularisation param for the lowest error
            best_score["indexb"]: degree param for the lowest error
    res_test : np.array
        Individual results for the test-set computations in dims od (nparamsA * nparamsB, 3)
            results[0, :]: indexA
            results[1, :]: indexB
            results[2, :]: score
    bs_test : dictionary
        Dictionary with values of best results for the test-set study:
            best_score["value"]: lowest error
            best_score["indexa"]: regularisation param for the lowest error
            best_score["indexb"]: degree param for the lowest error
    use_test_set : bool
        If true, test_set_cross_validate_hyperparameters() was used to obtain the values
    """

    for idx, value in enumerate(valuesB):
        start = len(valuesA) * idx
        end = len(valuesA) * (idx + 1)
        plotname = "{}_reg_for_{}_degree".format(name, value)

        if use_test_set:
            plotname = "testset_" + plotname
            data1 = dict(x=res[start: end, 0],
                         y=res[start: end, 2],
                         yerror=res[start: end, 3],
                         data_label="Results on subset of the test set")
        else:
            data1 = dict(x=res[start: end, 0],
                         y=res[start: end, 2],
                         yerror=res[start: end, 3],
                         data_label="Results from cross-validation")
        data2 = dict(x=res_test[start: end, 0],
                     y=res_test[start: end, 2],
                     yerror=np.zeros_like(res_test[start: end, 2]),
                     data_label="Results on the test set")

        if name == "band_gap":
            ylim = (0.07, 0.12)
        else:
            ylim = (0.015, 0.03)
        kt.plot_data_multiple([data1, data2],
                              plotname,
                              xlabel="Regularisation parameter",
                              ylabel="RMSL error",
                              fontsize=14,
                              ylim=ylim,
                              logplot=True)


def cross_validation_parameters_search(use_test_set=False):
    """Use cross validation for parameters search"""
    names = ["formation_energy", "band_gap"]
    valuesA = [1e-6 * (1.5 ** idx) for idx in range(20)]
    # valuesA = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    valuesB = [1, 2, 3, 4, 6]
    n_iter = 100

    for name in names:
        print("\nPredicting for", name)
        train_dataset, test_dataset = load_dataset(name, split=False)

        # ------ Cross validate parameters
        if use_test_set:
            bs, res = test_set_cross_validate_hyperparameters(train_dataset, test_dataset,
                                                              valuesA, valuesB,
                                                              300, n_iter)
        else:
            bs, res = cross_validate_hyperparameters(train_dataset,
                                                     valuesA, valuesB,
                                                     2000, 386, n_iter)

        # ------ Test-set best parameters
        bs_test, res_test = optimize_hyperparameters(train_dataset.descriptors,
                                                     train_dataset.properties,
                                                     test_dataset.descriptors,
                                                     test_dataset.properties,
                                                     valuesA,
                                                     valuesB)

        # Plot results
        plot_hyperparameters_search_results(name, valuesA, valuesB,
                                            res, bs, res_test, bs_test, use_test_set)
        # ------ Result with the best parameters from the cv set
        print("Best values:", bs)
        single_test(name, bs["indexa"], bs["indexb"])

        # ------ Dump the results
        filename = name + "_data.pkl"
        if use_test_set:
            filename = "test" + filename

        with open(filename, "wb") as fp:
            pickle.dump((bs, res, bs_test, res_test), fp)


def redo_plots_from_data(use_test_set=False):
    """Use pickled data to redo the plots"""
    names = ["formation_energy", "band_gap"]
    valuesA = [1e-6 * (1.5 ** idx) for idx in range(20)]
    # valuesA = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    valuesB = [1, 2, 3, 4, 6]

    for name in names:
        print("\nReplotting for", name)
        # ------ load the results
        filename = name + "_data.pkl"
        if use_test_set:
            filename = "test" + filename

        with open(filename, "rb") as fp:
            bs, res, bs_test, res_test = pickle.load(fp)

        # Plot results
        plot_hyperparameters_search_results(name, valuesA, valuesB,
                                            res, bs, res_test, bs_test, use_test_set)


def run_test_set_hyperparameters_search():
    """Use the whole test set to optimise hyperparameters"""

    names = ["formation_energy", "band_gap"]
    valuesA = [1e-6 * (1.5 ** idx) for idx in range(20)]
    # valuesA = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    valuesB = [1, 2, 3, 4, 6]

    for name in names:
        print("\nPredicting for", name)
        x_train, y_train, x_test, y_test = load_dataset(name, split=True)
        bs, res = optimize_hyperparameters(x_train, y_train, x_test, y_test, valuesA, valuesB, verbose=False)
        print("Best value:", bs)
        single_test(name, bs["indexa"], bs["indexb"])


def main():
    # cross_validation_parameters_search()
    # cross_validation_parameters_search(use_test_set=True)

    # redo_plots_from_data()
    # redo_plots_from_data(use_test_set=True)

    # run_test_set_hyperparameters_search()
    # single_test("formation_energy", 1e-5, 2)
    # single_test("band_gap", 1e-5, 4)
    """
    OPTIMAL PARAMS:
    Formation energy
        Polynomial Kernel (noide, degree)
        {'value': 0.021665742192413927, 'indexa': 1e-05, 'indexb': 1}
        Polynomial Kernel optimised on test set
        {'value': 0.019762374990609007, 'indexa': 1e-05, 'indexb': 2}

    Band GAP:
        Polynomial Kernel best value (noise, degree)
        {'value': 0.079997356976277773, 'indexa': 0.0005, 'indexb': 1}
        Polynomial kernel optimised on test set
        {'value': 0.0789, 'indexa': 1e-05, 'indexb': 4}
    """


if (__name__ == "__main__"):
    main()
