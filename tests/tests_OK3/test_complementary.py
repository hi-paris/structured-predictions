"""
Complementary testing for OK3
"""
import time
import pytest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error as MSE
# from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
# from sklearn.utils.estimator_checks import check_estimator
from stpredictions.models.OK3._classes import OK3Regressor
from stpredictions.models.OK3._classes import ExtraOK3Regressor

# Estimators to check
OKTREES = {
    "OK3Regressor": OK3Regressor,
    "ExtraOK3Regressor": ExtraOK3Regressor
}

# Datasets used
diabetes = load_diabetes(return_X_y=True)
iris = load_iris(return_X_y=True)
digits = load_digits(return_X_y=True)

# Datasets assemble
DATASETS = {
    "diabetes": {'X': diabetes[0], 'Y': diabetes[1]},
    "iris": {'X': iris[0], 'Y': iris[1]},
    "digits": {'X': digits[0], 'Y': digits[1]},
    # "bibtex": {'X': bibtex[0], 'Y': bibtex[1]},
    # "corel5K": {'X': corel5k[0], 'Y': corel5k[1]},
}


# Validation of the datasets
@pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
def test_X_y(nameSet, dataXY):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D.
    By default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y,
    such as checking that y does not have np.nan or np.inf targets.

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
    check_X_y(dataXY["X"], dataXY["Y"])


def fitted_predicted_OK3(X, y, Tree):
    """Function running OK3 and returning Y_train, Y_test, and Y_preds, for readability purposes

    Parameters
    ----------
    X : np.ndarray
        features of the dataset
    y : np.ndarray
        labels of the dataset
    Tree : estimator
        estimator to fit and predict
    Returns
    -------
    results : dict
        contains 'Y_train', 'Y_test', 'Y_pred_test'
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    reg = Tree()
    reg.fit(X_train, Y_train)
    # Y_pred_train = reg.predict(X_train, Y_train)
    Y_pred_test = reg.predict(X_test)
    results = {'Y_train': Y_train,
               'Y_test': Y_test,
               # 'Y_pred_train': Y_pred_train,
               'Y_pred_test': Y_pred_test}
    return results


class TestFit():
    """
    Test class for the .fit() function
    """

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    def test_time(self, nameTree, Tree, nameSet, dataXY):
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

        Returns
        -------
        None
        """
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY['X'], dataXY['Y'], test_size=0.33, random_state=42)
        reg = Tree()
        t0 = time.time()
        reg.fit(X_train, Y_train)
        fit_time = time.time() - t0
        assert fit_time < 100, f"Failed with {nameTree}/{nameSet}: 'fit_time' is over 100 seconds"

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
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
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY['X'], dataXY['Y'], test_size=0.33, random_state=42)
        reg = Tree()
        with pytest.raises(NotFittedError):
            check_is_fitted(reg)
        reg.fit(X_train, Y_train)
        assert check_is_fitted(reg) is None, f"Failed with {nameTree}/{nameSet}: 'reg' should be fitted after fitting"


class TestPredict():
    """
    Test class for the .predict() function
    """

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    def test_model_return_object(self, nameTree, Tree, nameSet, dataXY):
        """Checks that our fitted_predicted_estimator returns the good object

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
        returned_OK3 = fitted_predicted_OK3(dataXY['X'], dataXY['Y'], Tree)
        # Check the return object type
        assert isinstance(returned_OK3, dict), f'Failed with {nameTree}/{nameSet}:' \
                                               f'"returned_IOKR" should be a dict, ' \
                                               f'but instead is a {type(returned_OK3)}'
        # Check the length of the returned object
        assert len(returned_OK3) == 3, f'Failed with {nameTree}/{nameSet}:' \
                                       f'"returned_IOKR" should have a length of 3, ' \
                                       f'but instead it is {len(returned_OK3)}'

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    def test_model_return_values(self, nameTree, Tree, nameSet, dataXY):
        """Checks that 'Y_train', 'Y_test', 'Y_pred_test' from our fitted_predicted_estimator are
        non-empty, and np.ndarray

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
        returned_OK3 = fitted_predicted_OK3(dataXY['X'], dataXY['Y'], Tree)
        # Check returned arrays' type
        assert returned_OK3['Y_train'].size != 0, f"Failed with {nameTree}/{nameSet}:'Y_train' is empty"
        assert returned_OK3['Y_test'].size != 0, f"Failed with {nameTree}/{nameSet}:'Y_test' is empty"
        assert returned_OK3['Y_pred_test'].size != 0, f"Failed with {nameTree}/{nameSet}:'Y_pred_test' is empty"
        assert isinstance(returned_OK3['Y_train'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_train' should be a 'np.ndarray, but is {type(returned_OK3['Y_train'])}"
        assert isinstance(returned_OK3['Y_test'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_test' should be a 'np.ndarray, but is {type(returned_OK3['Y_test'])}"
        assert isinstance(returned_OK3['Y_pred_test'], np.ndarray), \
            f"Failed with {nameTree}/{nameSet}:'Y_pred_test' should be a 'np.ndarray, but is {type(returned_OK3['Y_pred_test'])}"

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
    def test_time(self, nameTree, Tree, nameSet, dataXY):
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

        Returns
        -------
        None
        """
        X_train, X_test, Y_train, Y_test = train_test_split(dataXY['X'], dataXY['Y'], test_size=0.33, random_state=42)
        reg = Tree()
        reg.fit(X_train, Y_train)
        test_t0 = time.time()
        Y_pred_test = reg.predict(X=X_test)
        test_pred_time = time.time() - test_t0
        assert test_pred_time < 100, f'Failed with {nameTree}/{nameSet}:"test_pred_time" is over 100 seconds'

    @pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
    @pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
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
        fp_0K3 = fitted_predicted_OK3(dataXY['X'], dataXY['Y'], Tree)
        test_mse = MSE(fp_0K3['Y_test'], fp_0K3['Y_pred_test'])
        test_rmse = test_mse ** 0.5
        threshold = 100000
        assert test_mse < threshold, f'Failed with {nameTree}/{nameSet}: mse = {test_mse}, but threshold set to {threshold} '
        assert test_rmse < threshold, f'Failed with {nameTree}/{nameSet}: rmse = {test_rmse}, but threshold set to {threshold} '


@pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
@pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
def test_bad_X_y_inputation(nameTree, Tree, nameSet, dataXY):
    """Checks the reaction of the estimator with bad inputation of X and y

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
    # ValueError
    with pytest.raises(ValueError):
        # Insert a np.nan into the X array
        Xt, yt = dataXY['X'], dataXY['Y']
        Xt[1] = np.nan
        scores = fitted_predicted_OK3(Xt, yt, Tree)
    with pytest.raises(ValueError):
        # Insert a np.nan into the y array
        Xt, yt = dataXY['X'], dataXY['Y']
        yt[1] = np.nan
        scores = fitted_predicted_OK3(Xt, yt, Tree)

    with pytest.raises(ValueError) as exception:
        # Insert a string into the X array
        Xt, yt = dataXY['X'], dataXY['Y']
        Xt[1] = "A string"
        scores = fitted_predicted_OK3(Xt, yt, Tree)
        assert "could not convert string to float" in str(exception.value)

    # Test that it handles the case of: X is a string
    with pytest.raises(ValueError):
        msg = fitted_predicted_OK3('X', dataXY['Y'], Tree)
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == "X must be a Numpy array"
    # Test that it handles the case of: y is a string
    with pytest.raises(ValueError):
        msg = fitted_predicted_OK3(dataXY['X'], 'y', Tree)
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == "y must be a Numpy array"


@pytest.mark.parametrize("nameTree, Tree", OKTREES.items())
@pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())
def test_bad_argument_inputation(nameTree, Tree, nameSet, dataXY):
    """Checks the reaction of the estimator with bad inputation of other arguments

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
    X_train, _, Y_train, _ = train_test_split(dataXY['X'], dataXY['Y'], test_size=0.33, random_state=42)
    reg = Tree()
    with pytest.raises(ValueError):
        msg = reg.fit(X_train, Y_train, check_input='True')
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == " must be a Boolean"
    with pytest.raises(ValueError):
        msg = reg.fit(X_train, Y_train, in_ensemble='False')
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == " must be a Boolean"
    with pytest.raises(ValueError):
        msg = reg.fit(X_train, Y_train, kernel=1)
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == " must be a str"

# def test_sklearn_check_estimator():
#    """test with check_estimator from sklearn"""
#    for name, Tree in OKTREES.items():
#        check_estimator(Tree())

# from os.path import join
# from sklearn.utils._testing import assert_array_equal
# from sklearn.utils._testing import assert_array_almost_equal
# from sklearn.utils._testing import assert_almost_equal
