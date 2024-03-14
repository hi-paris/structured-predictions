######## Status ###########
## File is executed withou error
## use below command
## python tests/tests_OK3/test_classification_forest.py


# import numpy as np
import time

from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from stpredictions.models.OK3._forest import RandomOKForestRegressor

n_train = 1000  # 1500 max sinin ordi freeze
n_test = 5000
n_features = 15

X_multilabel, y_multilabel = datasets.make_multilabel_classification(n_samples=n_train + n_test, n_features=n_features)

X_multilabel_train = X_multilabel[:n_train]
X_multilabel_test = X_multilabel[n_train:n_train + n_test]

y_multilabel_train = y_multilabel[:n_train]
y_multilabel_test = y_multilabel[n_train:n_train + n_test]

X_blobs, y_blobs = make_blobs(n_samples=n_train + n_test, n_features=n_features, centers=10)

X_blobs_train = X_blobs[:n_train]
X_blobs_test = X_blobs[n_train:n_train + n_test]

y_blobs_train = y_blobs[:n_train]
y_blobs_test = y_blobs[n_train:n_train + n_test]



def test_classification_multilabel(X_train, y_train, X_test, y_test):
    """Return the test accuracy for a multiclass classification task"""

    clf = RandomOKForestRegressor(oob_score=True, kernel="gini_clf")

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("base_estimator_ :\n", clf.base_estimator_)
    print("estimators_[0] :", clf.estimators_[0])
    print("feature_importances_ :", clf.feature_importances_)
    print("n_features_in :\n", clf.n_features_in_)
    print("n_outputs_ :", clf.n_outputs_)
    if hasattr(clf, "oob_score_"):
        print("oob_score_ (r2 score in HS):", clf.oob_score_)
    if hasattr(clf, "oob_decoded_score_"):
        print("oob_decoded_score_ (accuracy):", clf.oob_decoded_score_)

    y_pred = clf.predict(X_train)

    train_acc = 0
    for i in range(len(y_train)):
        train_acc += (y_pred[i] == y_train[i]).all()
    train_acc /= len(y_train)
    print()
    print("train accuracy :", train_acc)

    train_f1 = f1_score(y_train, y_pred, average='macro')
    print("train f1 score :", train_f1)

    train_hamming_score = (y_pred == y_train).mean()
    print("train hamming score :", train_hamming_score)

    y_pred = clf.predict(X_test)

    test_acc = 0
    for i in range(len(y_test)):
        test_acc += (y_pred[i] == y_test[i]).all()
    test_acc /= len(y_test)
    print()
    print("test accuracy :", test_acc)

    test_f1 = f1_score(y_test, y_pred, average='macro')
    print("test f1 score :", test_f1)

    test_hamming_score = (y_pred == y_test).mean()
    print("test hamming score :", test_hamming_score)

    return clf


def test_classification_multilabel_ref(X_train, y_train, X_test, y_test):
    """Return the test accuracy for a multiclass classification task"""

    clf = RandomForestClassifier(oob_score=True)

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("base_estimator_ :\n", clf.base_estimator_)
    print("estimato'RandomForestClassifier' object has no attribute 'n_features_'rs_[0] :", clf.estimators_[0])
    print("feature_importances_ :", clf.feature_importances_)
    print("n_features_in :\n", clf.n_features_in_)
    print("n_outputs_ :", clf.n_outputs_)
    if hasattr(clf, "oob_score_"):
        print("oob_score_ (accuracy):", clf.oob_score_)

    y_pred = clf.predict(X_train)

    train_acc = 0
    for i in range(len(y_train)):
        train_acc += (y_pred[i] == y_train[i]).all()
    train_acc /= len(y_train)
    print()
    print("train accuracy :", train_acc)

    train_f1 = f1_score(y_train, y_pred, average='macro')
    print("train f1 score :", train_f1)

    train_hamming_score = (y_pred == y_train).mean()
    print("train hamming score :", train_hamming_score)

    y_pred = clf.predict(X_test)

    test_acc = 0
    for i in range(len(y_test)):
        test_acc += (y_pred[i] == y_test[i]).all()
    test_acc /= len(y_test)
    print()
    print("test accuracy :", test_acc)

    test_f1 = f1_score(y_test, y_pred, average='macro')
    print("test f1 score :", test_f1)

    test_hamming_score = (y_pred == y_test).mean()
    print("test hamming score :", test_hamming_score)

    return clf


print("\n********************************")
print("********** MULTILABEL **********")
print("********************************")

print("\n----- Decision tree -----\n")
clf_ref_1 = test_classification_multilabel_ref(X_multilabel_train, y_multilabel_train, X_multilabel_test,
                                               y_multilabel_test)
print("\n----- OK3 -----\n")
clf_1 = test_classification_multilabel(X_multilabel_train, y_multilabel_train, X_multilabel_test, y_multilabel_test)

print("\n***************************")
print("********** BLOBS **********")
print("*****************************")

print("\n----- Decision tree -----\n")
clf_ref_2 = test_classification_multilabel_ref(X_blobs_train, y_blobs_train, X_blobs_test, y_blobs_test)
print("\n----- OK3 -----\n")
clf_2 = test_classification_multilabel(X_blobs_train, y_blobs_train, X_blobs_test, y_blobs_test)
