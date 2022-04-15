"""
Tests for model/model.py
Documentation in progress 
"""

import pytest
from stpredictions.data.load_data_IOKR import load_bibtex, load_corel5k
from stpredictions.IOKR.utils import project_root
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_iris
from sklearn.utils.validation import check_is_fitted, check_X_y
import numpy as np
from sklearn.exceptions import NotFittedError
from stpredictions.IOKR.model import IOKR as iokr
import time
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal

# Estimators to check
ESTIMATORS = {
    "IOKR": iokr,
}

# Datasets used
iris = load_iris(return_X_y=True)

# Datasets used
bibtex = load_bibtex()
corel5k = load_corel5k()

DATASETS = {
    "bibtex": {'X': bibtex[0], 'Y': bibtex[1]},
    "corel5K": {'X': corel5k[0], 'Y': corel5k[1]},
    # "iris": {'X': iris[0], 'Y': iris[1]},
}


# @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
# def test_X_y(nameSet, dataXY):
#    """Input validation for standard estimators.
#
#    Checks X and y for consistent length, enforces X to be 2D and y 1D.
#    By default, X is checked to be non-empty and containing only finite values.
#    Standard input checks are also applied to y,
#    such as checking that y does not have np.nan or np.inf targets.
#
#    Returns
#    -------
#    None
#    """
#    check_X_y(dataXY["X"], dataXY["Y"])


def fitted_predicted_IOKR(X, y, L=1e-5, ):
    """Function running IOKR and returning Y_train, Y_test, and Y_preds, for readability purposes

    Parameters
    ----------
    X : np.ndarray
        features of the dataset
    y : np.ndarray
        labels of the dataset
    L :

    Returns
    -------
    results : dict
        contains 'Y_train', 'Y_test', 'Y_pred_test'
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = iokr()
    clf.verbose = 1
    clf.fit(X=X_train, Y=Y_train, L=L)
    Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_test)
    results = {'Y_train': Y_train,
               'Y_test': Y_test,
               'Y_pred_test': Y_pred_test}
    return results


class TestFit:
    """
    class for the tests concerning .fit()
    """

    def test_verbose(self, capfd):
        """Test if fit function actually prints something

        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.

        Returns
        -------
        None
        """

        """Test if fit function actually prints something"""
        scores = fitted_predicted_IOKR(DATASETS['bibtex']['X'], DATASETS['bibtex']['Y'], L=1e-5)
        out, err = capfd.readouterr()
        assert err == "", f'{err}: need to be fixed'
        assert out != "", f'Fitting Time should have been printed '

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_time(self, nameTree, Tree, nameSet, dataXY, L=1e-5):
        """Test the time for fitting

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset
        L:

        Returns
        -------
        None
        """
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY['X'], dataXY['Y'], test_size=0.33, random_state=42)
        clf = Tree()
        clf.verbose = 1
        t0 = time.time()
        clf.fit(X=X_train, Y=Y_train, L=L, )
        fit_time = time.time() - t0
        assert fit_time < 100, f'Failed with {nameTree}/{nameSet}: "fit_time" is over 100 seconds'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_fit_fits(self, nameTree, Tree, nameSet, dataXY):
        """Checks that using the .fit() function actually fit the estimator

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY["X"], dataXY["Y"], test_size=0.33, random_state=42)
        clf1 = Tree()
        clf1.verbose = 1
        with pytest.raises(NotFittedError):
            check_is_fitted(clf1)
        clf2 = clf1
        clf2.fit(X=X_train, Y=Y_train, L=1e-5)
        assert np.array_equal(clf1, clf2), f"Failed with {nameTree}/{nameSet}: 'clf' should be fitted after fitting"

    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_numerical_stability(self, nameTree, Tree, L=1e-5):
        """Check numerical stability

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        L:

        Returns
        -------
        None
        """
        X = np.array([
            [152.08097839, 140.40744019, 129.75102234, 159.90493774],
            [142.50700378, 135.81935120, 117.82884979, 162.75781250],
            [127.28772736, 140.40744019, 129.75102234, 159.90493774],
            [132.37025452, 143.71923828, 138.35694885, 157.84558105],
            [103.10237122, 143.71928406, 138.35696411, 157.84559631],
            [127.71276855, 143.71923828, 138.35694885, 157.84558105],
            [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

        y = np.array(
            [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916, 0.49622521])

        with np.errstate(all="raise"):
            reg = Tree()
            reg.fit(X, y, L=L, )
            reg.fit(X, -y, L=L, )
            reg.fit(-X, y, L=L, )
            reg.fit(-X, -y, L=L, )

    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_raise_error_on_1d_input(self, nameTree, Tree, L=1e-5, ):
        """Test that an error is raised when X or Y are 1D arrays

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        L:

        Returns
        -------
        None
        """
        Xt = DATASETS['bibtex']['X'][:, 0].ravel()
        Xt_2d = DATASETS['bibtex']['X'][:, 0].reshape((-1, 1))
        yt = DATASETS['bibtex']['Y']

        with pytest.raises(ValueError):
            Tree().fit(Xt, yt, L=L, )

        Tree().fit(Xt_2d, yt, L=L, )
        with pytest.raises(ValueError):
            Tree().predict([Xt], yt)

#    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
#    def test_warning_on_big_input(self, nameTree, Tree, L=1e-5, ):
#        """Test if the warning for too large inputs is appropriate
#
#        Parameters
#        ----------
#        nameTree : str
#            Name of the tree estimator
#        Tree : estimator
#            estimator to check
#        L:
#
#        Returns
#        -------
#        None
#        """
#        Xt = np.repeat(10 ** 40., 4).astype(np.float64).reshape(-1, 1)
#        clf = Tree()
#        try:
#            clf.fit(Xt, [0, 1, 0, 1], L=L, )
#        except ValueError as e:
#            assert "float32" in str(e)


class TestPredict:

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_model_return_values(self, nameTree, Tree, nameSet, dataXY):
        """Tests for the returned values of the modeling function

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        returned_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        # Check returned arrays' type
        assert returned_IOKR['Y_train'].size != 0, "Failed with {name}:'Y_train' is empty"
        assert returned_IOKR['Y_test'].size != 0, "Failed with {name}:'Y_test' is empty"
        assert returned_IOKR['Y_pred_test'].size != 0, "Failed with {name}:'Y_pred_test' is empty"
        assert isinstance(returned_IOKR['Y_train'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_train' should be a 'np.ndarray, but is {type(returned_IOKR['Y_train'])}"
        assert isinstance(returned_IOKR['Y_test'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_test' should be a 'np.ndarray, but is {type(returned_IOKR['Y_test'])}"
        assert isinstance(returned_IOKR['Y_pred_test'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_pred_test' should be a 'np.ndarray, but is {type(returned_IOKR['Y_pred_test'])}"

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_model_return_object(self, nameTree, Tree, nameSet, dataXY):
        """Tests the returned object of the modeling function

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        returned_IOKR = fitted_predicted_IOKR(dataXY["X"], dataXY["Y"], L=1e-5, )
        # Check the return object type
        assert isinstance(returned_IOKR,
                          dict), f'Failed with {nameTree}/{nameSet}:' \
                                 f'"returned_IOKR" should be a dict, ' \
                                 f'but instead is a {type(returned_IOKR)}'
        # Check the length of the returned object
        assert len(
            returned_IOKR) == 3, f'Failed with {nameTree}/{nameSet}:' \
                                 f'"returned_IOKR" should have a length of 3, ' \
                                 f'but instead it is {len(returned_IOKR)}'

    def test_bad_X_y_inputation(self, ):
        """Tests for raised exceptions (To complete)

        Returns
        -------
        None
        """
        # ValueError
        with pytest.raises(ValueError):
            # Insert a np.nan into the X array
            Xt, yt = DATASETS['bibtex']['X'], DATASETS['bibtex']['Y']
            Xt[1] = np.nan
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, )
        with pytest.raises(ValueError):
            # Insert a np.nan into the y array
            Xt, yt = DATASETS['bibtex']['X'], DATASETS['bibtex']['Y']
            yt[1] = np.nan
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, )

        with pytest.raises(ValueError) as exception:
            # Insert a string into the X array
            Xt, yt = DATASETS['bibtex']['X'], DATASETS['bibtex']['Y']
            Xt[1] = "A string"
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, )
            assert "could not convert string to float" in str(exception.value)

        # Test that it handles the case of: X is a string
        with pytest.raises(ValueError):
            Xt, yt = DATASETS['bibtex']['X'], DATASETS['bibtex']['Y']
            msg = fitted_predicted_IOKR('Xt', yt, L=1e-5, )
            assert isinstance(msg, AssertionError)
            assert msg.args[0] == "X must be a Numpy array"
        # Test that it handles the case of: y is a string
        with pytest.raises(ValueError):
            Xt, yt = DATASETS['bibtex']['X'], DATASETS['bibtex']['Y']
            msg = fitted_predicted_IOKR(Xt, 'yt', L=1e-5, )
            assert isinstance(msg, AssertionError)
            assert msg.args[0] == "y must be a Numpy array"

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_time(self, nameTree, Tree, nameSet, dataXY, L=1e-5, ):
        """Test the time for predicting

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset
        L:

        Returns
        -------
        None
        """
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY["X"], dataXY["Y"], test_size=0.33, random_state=42)
        clf = Tree()
        clf.verbose = 1
        clf.fit(X=X_train, Y=Y_train, L=L, )
        test_t0 = time.time()
        Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_train)
        test_pred_time = time.time() - test_t0
        assert test_pred_time < 100, f'Failed with {nameTree}/{nameSet}:"test_pred_time" is over 100 seconds'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_recall(self, nameTree, Tree, nameSet, dataXY):
        """Test the recall score

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        fp_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        recall_test = recall_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'],average='micro')
        threshold = 0
        assert recall_test > threshold, f'Failed with {nameTree}/{nameSet}: recall_test = {recall_test},' \
                                        f'but threshold set to {threshold}'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_precision(self, nameTree, Tree, nameSet, dataXY):
        """Tests the precision score
        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        fp_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        precision_test = precision_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'],average='micro')
        threshold = 0
        assert precision_test > threshold, f'Failed with {nameTree}/{nameSet}: precision_test = {precision_test},' \
                                           f'but threshold set to {threshold}'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_f1_score(self, nameTree, Tree, nameSet, dataXY):
        """Tests the F1_score

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        fp_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        f1_test = f1_score(fp_IOKR['Y_pred_test'], fp_IOKR['Y_test'], average='samples')
        threshold = 0
        # Check f1 score range
        assert 1.0 >= f1_test >= 0.0, f"Failed with {nameTree}/{nameSet}: f1 score is of {f1_test}, should be between 0 and 1"
        # assert f1 score is enough
        assert f1_test > threshold, f'Failed with {nameTree}/{nameSet}: f1_test = {f1_test}, but threshold set to {threshold}'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_accuracy_score(self, nameTree, Tree, nameSet, dataXY):
        """Check accuracy of the model

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        fp_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        accuracy = accuracy_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        threshold = 0
        assert accuracy > threshold, f'Failed with {nameTree}/{nameSet}: accuracy = {accuracy}, but threshold set to {threshold}'

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    @pytest.mark.parametrize("nameTree, Tree", ESTIMATORS.items())
    def test_mse(self, nameTree, Tree, nameSet, dataXY):
        """Checks the MSE score of the model

        Parameters
        ----------
        nameTree : str
            Name of the tree estimator
        Tree : estimator
            estimator to check
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        fp_IOKR = fitted_predicted_IOKR(dataXY['X'], dataXY['Y'], L=1e-5, )
        test_mse = MSE(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        threshold = 0
        assert test_mse > threshold, f'Failed with {nameTree}/{nameSet}: mse = {test_mse}, but threshold set to {threshold} '


class TestAlpha:

    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    def test_alpha_returns(self, nameSet, dataXY):
        """Tests if the function returns what it should

        Parameters
        ----------
        nameSet : str
            Name of the dataset
        dataXY : tuple of arrays
            Containing X and y from the dataset

        Returns
        -------
        None
        """
        test_size = 0.33
        X_train, X_test, Y_train, Y_test = train_test_split(DATASETS['bibtex']['X'], DATASETS['bibtex']['Y'],
                                                            test_size=test_size, random_state=42)
        clf = iokr()
        clf.fit(X_train, Y_train, L= 1e-5)
        Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_test)
        A = clf.alpha(X_test)
        assert A is not None, f"A is None"
        assert A != "", f"A is empty"
        assert isinstance(A, np.ndarray), f"Failed with {nameSet}: A should be 'np.ndarray', but is {type(A)}"

# To confirm
# def test_sklearn_check_estimator():
#    """test with check_estimator from sklearn"""
#    check_estimator(iokr())

# To confirm
# def test_sklearn_check_estimator():
#    """test with check_estimator from sklearn"""
#    check_estimator(iokr())
