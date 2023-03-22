# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 12, 2021
"""

import arff
import numpy as np
import os
from os.path import join
from scipy import sparse
from stpredictions.models.DIOKR.utils import project_root

# bibtex
# files (sparse): Train and test sets along with their union and the XML header [bibtex.rar]
# source: I. Katakis, G. Tsoumakas, I. Vlahavas, "Multilabel Text Classification for Automated Tag Suggestion",
# Proceedings of the ECML/PKDD 2008 Discovery Challenge, Antwerp, Belgium, 2008.


# split dataset using
#
# dir_path = os.path.dirname(os.path.realpath(__file__))


# @profile

def load_bibtex():
    """
    Load the bibtex dataset for IOKR.
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

    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "bibtex", "bibtex.arff")

    feature_idx = 1836

    dataset = arff.load(open(DATA_PATH), "r")
    data = np.array(dataset['data'], np.int64)

    X = data[:, 0:feature_idx]
    Y = data[:, feature_idx:]

    return X, Y


def load_corel5k():
    """
    Load the corel5k dataset for IOKR.
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
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "corel5k", "Corel5k.arff")

    feature_idx = 499

    dataset = arff.load(open(DATA_PATH), "r")
    data = np.array(dataset['data'], np.int64)

    X = data[:, 0:feature_idx]
    Y = data[:, feature_idx:]

    X_txt = [t[0] for t in dataset['attributes'][:feature_idx]]
    Y_txt = [t[0] for t in dataset['attributes'][feature_idx:]]

    return X, Y, X_txt, Y_txt


def load_from_arff(filename, label_count, label_location="end",
                   input_feature_type='float', encode_nominal=True, load_sparse=False,
                   return_attribute_definitions=False):
    """Method for loading ARFF files as numpy array
    Parameters
    ----------
    filename : str
        path to ARFF file
    labelcount: integer
        number of labels in the ARFF file
    endian: str {"big", "little"} (default is "big")
        whether the ARFF file contains labels at the beginning of the
        attributes list ("start", MEKA format)
        or at the end ("end", MULAN format)
    input_feature_type: numpy.type as string (default is "float")
        the desire type of the contents of the return 'X' array-likes,
        default 'i8', should be a numpy type,
        see http://docs.scipy.org/doc/numpy/user/basics.types.html
    encode_nominal: bool (default is True)
        whether convert categorical data into numeric factors - required
        for some scikit classifiers that can't handle non-numeric
        input features.
    load_sparse: boolean (default is False)
        whether to read arff file as a sparse file format, liac-arff
        breaks if sparse reading is enabled for non-sparse ARFFs.
    return_attribute_definitions: boolean (default is False)
        whether to return the definitions for each attribute in the
        dataset
    Returns
    -------
    X : :mod:`scipy.sparse.lil_matrix` of `input_feature_type`, shape=(n_samples, n_features)
        input feature matrix
    y : :mod:`scipy.sparse.lil_matrix` of `{0, 1}`, shape=(n_samples, n_labels)
        binary indicator matrix with label assignments
    names of attributes : List[str]
        list of attribute names from ARFF file
    """

    if not load_sparse:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.DENSE
        )
        matrix = sparse.csr_matrix(
            arff_frame['data'], dtype=input_feature_type
        )
    else:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.COO
        )
        data = arff_frame['data'][0]
        row = arff_frame['data'][1]
        col = arff_frame['data'][2]
        matrix = sparse.coo_matrix(
            (data, (row, col)), shape=(max(row) + 1, max(col) + 1)
        )

    if label_location == "start":
        X, y = matrix.tocsc()[:, label_count:].tolil(), matrix.tocsc()[:, :label_count].astype(int).tolil()
        feature_names = arff_frame['attributes'][label_count:]
        label_names = arff_frame['attributes'][:label_count]
    elif label_location == "end":
        X, y = matrix.tocsc()[:, :-label_count].tolil(), matrix.tocsc()[:, -label_count:].astype(int).tolil()
        feature_names = arff_frame['attributes'][:-label_count]
        label_names = arff_frame['attributes'][-label_count:]
    else:
        # unknown endian
        return None

    if return_attribute_definitions:
        return X, y, feature_names, label_names
    else:
        return X, y

def load_bibtex_train_from_arff():
    """Load the bibtex dataset for DIOKR

    """
    path_tr = join(project_root(), 'datasets/bibtex/bibtex-train.arff')
    # print(path_tr)
    X_train, Y_train = load_from_arff(path_tr, label_count=159)
    return X_train, Y_train

def load_bibtex_test_from_arff():
    """Load the bibtex dataset for DIOKR

    """
    path_tr = join(project_root(), 'datasets/bibtex/bibtex-test.arff')
    # print(path_tr)
    X_test, Y_test = load_from_arff(path_tr, label_count=159)
    return X_test, Y_test

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

