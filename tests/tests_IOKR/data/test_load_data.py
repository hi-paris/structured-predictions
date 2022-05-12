"""Test module for the module: load_data.py"""

import numpy as np
from stpredictions.datasets.load_data import load_bibtex
from stpredictions.datasets.load_data import load_corel5k


class TestLoadBibtex():
    """Test class for the function: load_bibtex"""

    def test_returned_variables_not_empty(self):
        """Test checking if returned variables from load_bibtex are not empty

        Returns
        -------
        None
        """
        load = load_bibtex()
        print(load)
        assert load[0] is not None, "Expected variable: 'X'"
        assert load[1] is not None, "Expected variable: 'Y'"


    def test_returned_variables_good_type(self):
        """Test checking if returned variables from load_bibtex are the expected type

        Returns
        -------
        None
        """
        load = load_bibtex()
        actual_x = type(load[0])
        actual_y = type(load[1])
        expected1 = "np.array"
        assert isinstance(load[0], np.ndarray), f"'X' should be {expected1}, but is {actual_x} "
        assert isinstance(load[1], np.ndarray), f"'Y' should be {expected1}, but is {actual_y} "


    def test_returned_variables_good_shape(self):
        """Test checking if returned variables from load_bibtex are the expected shape

        Returns
        -------
        None
        """
        load = load_bibtex()
        actual_x_shape = load[0].shape
        actual_y_shape = load[1].shape
        expected_x_shape = (7395, 1836)
        expected_y_shape = (7395, 159)
        assert actual_x_shape == expected_x_shape, f"'X' should be {expected_x_shape}, but is {actual_x_shape} "
        assert actual_y_shape == expected_y_shape, f"'Y' should be {expected_y_shape}, but is {actual_y_shape} "


#    def test_check_X_y(self):
#        """Input validation for standard estimators.
#
#        Checks X and y for consistent length, enforces X to be 2D and y 1D.
#        By default, X is checked to be non-empty and containing only finite values.
#        Standard input checks are also applied to y,
#        such as checking that y does not have np.nan or np.inf targets.
#
#        Returns
#        -------
#        None
#        """
#        load = load_bibtex("IOKR/data/bibtex")
#        check = check_X_y(load[0], load[1])
#        assert check


class TestLoadCorel5k():
    """Test class for the function: load_corel5k"""

    def test_returned_variables_not_empty(self):
        """Test checking if returned variables from load_corel5k are not empty

        Returns
        -------
        None
        """
        load = load_corel5k()
        print(load)
        assert load[0] is not None, "Expected variable: 'X'"
        assert load[1] is not None, "Expected variable: 'Y'"
        assert load[2] != "", "Expected variable: 'X_txt'"
        assert load[3] != "", "Expected variable: 'Y_txt'"

    def test_returned_variables_good_type(self):
        """Test checking if returned variables from load_corel5k are the expected type

        Returns
        -------
        None
        """
        load = load_corel5k()
        actual_x = type(load[0])
        actual_y = type(load[1])
        actual_x_txt = type(load[2])
        actual_y_txt = type(load[3])
        expected1 = "np.array"
        expected2 = 'list'
        print(actual_x, actual_y, actual_x_txt, actual_y_txt)
        assert isinstance(load[0], np.ndarray), f"'X' should be {expected1}, but is {actual_x} "
        assert isinstance(load[1], np.ndarray), f"'Y' should be {expected1}, but is {actual_y} "
        assert isinstance(load[2], list), f"'X_txt' should be {expected2}, but is {actual_x_txt} "
        assert isinstance(load[3], list), f"'Y_txt' should be {expected2}, but is {actual_y_txt} "

    def test_returned_variables_good_shape(self):
        """Test checking if returned variables from load_corel5k are of expected shape

        Returns
        -------
        None
        """
        load = load_corel5k()
        actual_x_shape = load[0].shape
        actual_y_shape = load[1].shape
        actual_x_txt_len = len(load[2])
        actual_y_txt_len = len(load[3])
        expected_x_shape = (5000, 499)
        expected_y_shape = (5000, 374)
        expected_x_txt_len = 499
        expected_y_txt_len = 374
        print(actual_x_shape, actual_y_shape, actual_x_txt_len, actual_y_txt_len)
        assert actual_x_shape, f"'X' should be {expected_x_shape}, but is {actual_x_shape} "
        assert actual_y_shape, f"'Y' should be {expected_y_shape}, but is {actual_y_shape} "
        assert actual_x_txt_len, f"'X_txt' should be {expected_x_txt_len}, but is {actual_x_txt_len} "
        assert expected_y_txt_len, f"'Y_txt' should be {expected_y_txt_len}, but is {actual_y_txt_len} "
