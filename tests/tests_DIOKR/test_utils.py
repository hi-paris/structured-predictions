"""Test module for model/utils.py"""
# import pytest
# from os import listdir
# from os.path import isdir, isfile, dirname, join
# from stpredictions.models.DIOKR.utils import project_root

# BROKEN
# class TestProjectRoot():
#    """Test class for function project_root"""
#
#    def test_project_root(self):
#        """Test for project_root function"""
#        # Get the actual path
#        actual_path = dirname(__file__)
#        # Get the supposed root_path to check
#        root_path = dirname(project_root())
#        # Get the difference in length of both
#        a = len(actual_path)
#        b = len(root_path)
#        dif_length = a - b
#        # Subtract the difference to the actual path
#        root_actual_path = actual_path[:-dif_length]
#        # Assert that the root path from the function is the same as the root of the actual path
#        assert root_actual_path.lower() == root_path.lower(), \
#            f'Root_path should be {root_path}, but instead returned {root_actual_path}'
