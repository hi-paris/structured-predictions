# Implementation
import time
# from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
# from sklearn.metrics import f1_score
import numpy as np
# import pandas as pd
# from stpredictions.datasets.load_data import load_bibtex
# from sklearn.model_selection import train_test_split
# import arff
from numpy.linalg import inv
# import os

# from line_profiler import LineProfiler


# insert at position 1 in the path, as 0 is the path of this file.

# dir_path = os.path.dirname("/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex")
# dir_path = os.path.dirname(os.path.realpath(__file__))
# dataset = arff.load(open('/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex/bibtex.arff'), "r")


"""
Created on December 12, 2021
"""


class IOKR:
    """
    Class used to apply Input and Output Kernel Regression
    """

    #    @profile
    def __init__(self):
        """
        Initialization of the below parameters.

        Parameters
        ----------
        X_train :  sparse matrix - containing explanatory variables of the train set
        Y_train: sparse matrix - containing target variable of the train set
        Ky: output scalar kernel
        M: gram matrix on training set
        verbose: binary - display more parameters
        linear:

        """
        self.X_train = None
        self.Y_train = None
        self.Ky = None
        self.M = None
        self.verbose = 0
        self.input_kernel = None
        self.output_kernel = None

    #    @profile
    def fit(self, X, Y, L, input_kernel='linear', input_kernel_param=None):
        """
        Model Fitting

        """

        # save input and output training data
        self.X_train, self.Y_train = X, Y

        # instantiate input kernel parameter when not given
        if input_kernel_param is None:
            if input_kernel == 'rbf':
                input_kernel_param = 1.
            elif input_kernel == 'polynomial':
                input_kernel_param = [3, None, 1]

        # define input kernel
        if input_kernel == 'linear':
            self.input_kernel = lambda A, B: linear_kernel(A, B)
        elif input_kernel == 'polynomial':
            self.input_kernel = lambda A, B: polynomial_kernel(A, B, degree=input_kernel_param[0],
                                                               gamma=input_kernel_param[1], coef0=input_kernel_param[2])
        elif input_kernel == 'rbf':
            self.input_kernel = lambda A, B: rbf_kernel(A, B, gamma=input_kernel_param)
        else:
            self.input_kernel = input_kernel

        # compute input gram matrix
        Kx = self.input_kernel(X, X)

        # kernel ridge regression training computation: n x n matrix inversion
        t0 = time.time()
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * L * np.eye(n))
        if self.verbose > 0:
            print(f'Fitting time: {time.time() - t0} in s')

    def alpha(self, X_test):

        Kx = self.input_kernel(self.X_train, X_test)
        A = self.M.dot(Kx)

        return A

    #    @profile
    def predict(self, X_test, Y_candidates, output_kernel='linear', output_kernel_param=None):

        """
        Model Prediction

        """

        # instantiate output kernel parameter when not given
        if output_kernel_param is None:
            if output_kernel == 'rbf':
                output_kernel_param = 1.
            elif output_kernel == 'polynomial':
                output_kernel_param = [3, None, 1]

        # define output kernel
        if output_kernel == 'linear':
            self.output_kernel = lambda A, B: linear_kernel(A, B)
        elif output_kernel == 'polynomial':
            self.output_kernel = lambda A, B: polynomial_kernel(A, B, degree=output_kernel_param[0],
                                                                gamma=output_kernel_param[1],
                                                                coef0=output_kernel_param[2])
        elif output_kernel == 'rbf':
            self.output_kernel = lambda A, B: rbf_kernel(A, B, gamma=output_kernel_param)
        else:
            self.output_kernel = output_kernel

        # compute output gram matrix
        Ky = self.output_kernel(self.Y_train, Y_candidates)

        # compute prediction
        t0 = time.time()
        Alpha = self.alpha(X_test)
        scores = Ky.transpose().dot(Alpha)
        idx_pred = np.argmax(scores, axis=0)
        Y_pred = Y_candidates[idx_pred]
        if self.verbose > 0:
            print(f'Decoding time: {time.time() - t0} in s')

        return Y_pred


#### Example v1 Debugging


# path = "/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex"
# X, Y, _, _ = load_bibtex(path)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# clf = IOKR()
# clf.verbose = 1
# L = 1e-5
# sx = 1000
# sy = 10
#
# clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
# Y_pred_train = clf.predict(X_train=X_train)
# Y_pred_test = clf.predict(X_test=X_test)
# f1_train = f1_score(Y_pred_train, Y_train, average='samples')
# f1_test = f1_score(Y_pred_test, Y_test, average='samples')
#
# print(f'Train f1 score: {f1_train} / Test f1 score {f1_test}')


#### Example v2 Debugging


# from IOKR.model.model import IOKR
# from sklearn.model_selection import train_test_split
# from IOKR.data.load_data import load_bibtex
# from sklearn.metrics import f1_score
'''
path = "IOKR/data/bibtex"
X, Y, _, _ = load_bibtex(path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(Y_train.shape)
clf = IOKR()
clf.verbose = 1
L = 1e-5
sx = 1000
sy = 10
clf.fit(X=X_train, Y=Y_train, L=L)
Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_test)
f1_test = f1_score(Y_pred_test, Y_test, average='samples')
print( "Test f1 score:", f1_test)
'''
