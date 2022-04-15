"""Test module for model/utils.py"""

from os.path import isdir  # , dirname
from stpredictions.models.IOKR.utils import create_path_that_doesnt_exist  # , project_root


class TestCreatePathThatDoesntExist():
    """Test class for create_path_that_doesnt_exist function"""

    def test_path_dir_is_created(self, tmp_path):
        """test to check if the path dir is created"""
        d1 = tmp_path / "Testdir"
        # f1 = d1 / 'Check.txt'
        assert isdir(d1) is False, f'{d1} already exist'
        create_path_that_doesnt_exist(d1, "Check", "txt")
        assert isdir(d1) is True, f'{d1} has not been created'

# BROKEN
# class TestProjectRoot():
#    """Test class for function project_root"""
#
#    def test_project_root(self, ):
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
#        assert root_actual_path == root_path, \
#            f'Root_path should be {root_path}, but instead returned {root_actual_path}'
