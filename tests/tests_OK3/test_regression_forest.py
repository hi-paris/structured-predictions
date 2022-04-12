# compare le tree de régression avec OK3 avec noyau linéaire

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from stpredictions.models.OK3._forest import RandomOKForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
perm = np.random.permutation(y.size)
X = X[perm]
y = y[perm]
n_train = 1000  # 1500 max sinon ordi freeze
n_test = 10000
X_train = X[:n_train]
X_test = X[n_train:n_train + n_test]
y_train = y[:n_train]
y_test = y[n_train:n_train + n_test]


def test_regression_ref(X_train, y_train, X_test, y_test):
    """Return the mse for a classical regression task"""

    clf = RandomForestRegressor(oob_score=True)

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("base_estimator_ :\n", clf.base_estimator_)
    print("estimators_[0] :", clf.estimators_[0])
    print("feature_importances_ :", clf.feature_importances_)
    print("n_features_ :\n", clf.n_features_)
    print("n_outputs_ :", clf.n_outputs_)
    if hasattr(clf, "oob_score_"):
        print("oob_score_ (r2 score in output space):", clf.oob_score_)

    y_pred = clf.predict(X_train)

    train_mse = mean_squared_error(y_train, y_pred)
    print("train mse :", train_mse)

    y_pred = clf.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print("test mse :", test_mse)

    return clf


def test_regression(X_train, y_train, X_test, y_test):
    """Return the test accuracy for a multiclass classification task"""

    clf = RandomOKForestRegressor(oob_score=True, kernel="mse_reg")

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("base_estimator_ :\n", clf.base_estimator_)
    print("estimators_[0] :", clf.estimators_[0])
    print("feature_importances_ :", clf.feature_importances_)
    print("n_features_ :\n", clf.n_features_)
    print("n_outputs_ :", clf.n_outputs_)
    if hasattr(clf, "oob_score_"):
        print("oob_score_ (r2 score in HS):", clf.oob_score_)
    if hasattr(clf, "oob_decoded_score_"):
        print("oob_decoded_score_ (r2 score in output space):", clf.oob_decoded_score_)

    y_pred = clf.predict(X_train)

    train_mse = mean_squared_error(y_train, y_pred)
    print("train mse :", train_mse)

    y_pred = clf.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    print("test mse :", test_mse)

    return clf


print("\n----- Random Forest -----\n")
reg_ref = test_regression_ref(X_train, y_train, X_test, y_test)
print("\n----- Random OK Forest -----\n")
reg = test_regression(X_train, y_train, X_test, y_test)
