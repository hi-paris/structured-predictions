# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 12, 2021
"""

import arff
import os
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from line_profiler import LineProfiler
import random

## bibtex
### files (sparse): Train and test sets along with their union and the XML header [bibtex.rar]
### source: I. Katakis, G. Tsoumakas, I. Vlahavas, "Multilabel Text Classification for Automated Tag Suggestion",
### Proceedings of the ECML/PKDD 2008 Discovery Challenge, Antwerp, Belgium, 2008.


# split dataset using
#
# dir_path = os.path.dirname(os.path.realpath(__file__))


# @profile

def load_bibtex(dir_path: str):
    """
    Load the bibtex dataset.
    __author__ = "Michael Gygli, ETH Zurich"
    from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py

    Parameters
    ----------
    dir_path : string - containing location of bibtex.arff


    Returns
    -------
    X : np.array
        Explanatory variables - N * 1836 array variables in one vector - e.g. 'dependent', 'always'

    Y : np.array
        Target variables - N * 159 array variables in one vector - e.g. 'TAG_system', 'TAG_social_nets'

    X_txt : list
            Explanatory variables - N * 1836 list variables in one vector - e.g. 'dependent', 'always'

    Y_txt : list
            Target variables - N * 159 list variables in one vector - e.g. 'TAG_system', 'TAG_social_nets'

    """

    feature_idx = 1836

    dataset = arff.load(open(dir_path+"/bibtex.arff"), "r")
    data = np.array(dataset['data'], np.int64)

    X = data[:, 0:feature_idx]
    Y = data[:, feature_idx:]

    X_txt = [t[0] for t in dataset['attributes'][:feature_idx]]
    Y_txt = [t[0] for t in dataset['attributes'][feature_idx:]]

    return X, Y, X_txt, Y_txt


def load_corel5k(dir_path: str):
    """
    Load the bibtex dataset.
    __author__ = "Michael Gygli, ETH Zurich"
    from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py

    Parameters
    ----------
    dir_path : string - containing location of bibtex.arff


    Returns
    -------
    X : np.array
        Explanatory variables - N * 499 array variables in one vector - e.g. 'dependent', 'always'

    Y : np.array
        Target variables - N * 374 array variables in one vector - e.g. 'TAG_system', 'TAG_social_nets'

    X_txt : list
            Explanatory variables - N * 499 list variables in one vector - e.g. 'dependent', 'always'

    Y_txt : list
            Target variables - N * 374 list variables in one vector - e.g. 'TAG_system', 'TAG_social_nets'

    """

    feature_idx = 499

    dataset = arff.load(open(dir_path+"/Corel5k.arff"), "r")
    data = np.array(dataset['data'], np.int64)

    X = data[:, 0:feature_idx]
    Y = data[:, feature_idx:]

    X_txt = [t[0] for t in dataset['attributes'][:feature_idx]]
    Y_txt = [t[0] for t in dataset['attributes'][feature_idx:]]

    return X, Y, X_txt, Y_txt



# ####### Use Case
# path = "../data/bibtex"
# X, Y, _, _ = load_bibtex(path)
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#
#
# n_tr = X_train.shape[0]
# n_te = X_test.shape[0]
# input_dim = X_train.shape[1]
# label_dim = Y_train.shape[1]
#
# print(f'Train set size = {n_tr}')
# print(f'Test set size = {n_te}')
# print(f'Input dim. = {input_dim}')
# print(f'Output dim. = {label_dim}')
# print(len(Y_test))
# print(len(Y_train))
