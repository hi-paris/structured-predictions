# compare le tree de régression avec OK3 avec noyau linéaire

import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from stpredictions.models.OK3._classes import OK3Regressor
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
perm = np.random.permutation(y.size)
X = X[perm]
y = y[perm]
n_train = 5000
n_test = 5000
X_train = X[:n_train]
X_test = X[n_train:n_train + n_test]
y_train = y[:n_train]
y_test = y[n_train:n_train + n_test]


def test_regression_ref(X_train, y_train, X_test, y_test):
    """Return the mse for a classical regression task"""

    clf = DecisionTreeRegressor()

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(end_fit - start_fit) + " s.")

    print("first nodes impurities :\n", clf.tree_.impurity[:10])
    print("number of leaves :", clf.get_n_leaves())
    print("depth of the tree :", clf.get_depth())

    y_pred = clf.predict(X_train)

    train_mse = mean_squared_error(y_train, y_pred)
    print("train mse :", train_mse)

    y_pred = clf.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print("test mse :", test_mse)

    return clf


def test_regression(X_train, y_train, X_test, y_test):
    """Return the test accuracy for a multiclass classification task"""

    clf = OK3Regressor(kernel="mse_reg")

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(end_fit - start_fit) + " s.")

    print("first nodes impurities :\n", clf.tree_.impurity[:10])
    print("number of leaves :", clf.get_n_leaves())
    print("depth of the tree :", clf.get_depth())

    y_pred = clf.predict(X_train)

    train_mse = mean_squared_error(y_train, y_pred)
    print("train mse :", train_mse)

    y_pred = clf.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print("test mse :", test_mse)

    return clf


print("\n----- Decision tree -----\n")
reg_ref = test_regression_ref(X_train, y_train, X_test, y_test)
print("\n----- OK3 -----\n")
reg = test_regression(X_train, y_train, X_test, y_test)
