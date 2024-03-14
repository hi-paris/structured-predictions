######## Status ###########
## File is executed withou error
## use below command
## python tests/tests_OK3/test_classification.py



from stpredictions.models.OK3._classes import OK3Regressor
# from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import time

from stpredictions.models.OK3._classes import OK3Regressor

n_samples = 10000
n_features = 15

X_multilabel, y_multilabel = datasets.make_multilabel_classification(n_samples=n_samples, n_features=n_features)

X_multilabel_train = X_multilabel[:n_samples // 2]
X_multilabel_test = X_multilabel[n_samples // 2:]

y_multilabel_train = y_multilabel[:n_samples // 2]
y_multilabel_test = y_multilabel[n_samples // 2:]

X_blobs, y_blobs = datasets.make_blobs(n_samples=n_samples, n_features=n_features, centers=10)

X_blobs_train = X_blobs[:n_samples // 2]
X_blobs_test = X_blobs[n_samples // 2:]

y_blobs_train = y_blobs[:n_samples // 2]
y_blobs_test = y_blobs[n_samples // 2:]


def test_classification_multilabel(X_train, y_train, X_test, y_test):
    """Return the test accuracy for a multiclass classification task"""

    clf = OK3Regressor(kernel="gini_clf")

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("first nodes impurities :\n", clf.tree_.impurity[:5])
    print("number of leaves :", clf.get_n_leaves())
    print("depth of the tree :", clf.get_depth())

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

    clf = DecisionTreeClassifier()

    start_fit = time.time()

    clf.fit(X_train, y_train)

    end_fit = time.time()

    print("reference fitting time : " + str(1000 * (end_fit - start_fit)) + " ms")

    print("first nodes impurities :\n", clf.tree_.impurity[:5])
    print("number of leaves :", clf.get_n_leaves())
    print("depth of the tree :", clf.get_depth())

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
