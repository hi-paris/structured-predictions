from os.path import join

import pytest
import torch
from stpredictions.datasets.load_data import load_bibtex_train_from_arff, load_bibtex_test_from_arff

from stpredictions.models.DIOKR import kernel, cost
from stpredictions.models.DIOKR.kernel import linear_kernel, rbf_kernel, gaussian_tani_kernel
from stpredictions.models.DIOKR.utils import project_root

x_train, y_train = load_bibtex_train_from_arff()
x_train, y_train = x_train.todense(), y_train.todense()


x_test, y_test = load_bibtex_test_from_arff()
x_test, y_test = x_test.todense(), y_test.todense()

x_train, y_train = x_train[:2000], y_train[:2000]
x_test, y_test = x_test[:500], y_test[:500]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

n = x_train.shape[0]

gamma_input = 0.1
gamma_output = 1.0

kernel_output = kernel.Gaussian(gamma_output)
lbda = 0.01

batch_size_train = 256

cost_function = cost.sloss_batch

d_out = int(x_train.shape[1] / 2)
model_kernel_input = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], d_out),
    torch.nn.ReLU(),
)

X_kernel = torch.normal(0, 1, size=(3, 1))
Y_kernel = torch.normal(0, 1, size=(2, 1)),

optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)


class TestLinearKernel():
    "Class test for linear Kernel function"

    def test_output_good_type_and_exists(self):
        """Test that linear_kernel outputs exists with good type

        Returns
        -------
        None
        """
        K = linear_kernel(X_kernel)
        # print("XK:",X_kernel)
        # print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        """Tests that linear_kernel reacts well with bad inputation

        Returns
        -------
        None
        """
        # Test that it handles the case of: Xkernel is a string
        with pytest.raises(AttributeError):
            msg = linear_kernel('X_kernel')
            assert isinstance(msg, AttributeError)
        # Test that it handles the case of: Xkernel is None
        with pytest.raises(AttributeError):
            X_kernel = None
            msg = linear_kernel(X_kernel)
            assert isinstance(msg, AttributeError)


class TestRbfKernel():
    "Class test for rbf Kernel function"

    def test_output_good_type_and_exists(self):
        """Tests that rbf_kernel output exists with good type

        Returns
        -------
        None
        """
        K = rbf_kernel(X_kernel)
        # print("XK:",X_kernel)
        # print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        """Test how rbf_kernel reacts with bad inputation

        Returns
        -------
        None
        """
        # Test that it handles the case of: Xkernel is a string
        with pytest.raises(AttributeError):
            msg = rbf_kernel('X_kernel')
            assert isinstance(msg, AttributeError)
        # Test that it handles the case of: Xkernel is None
        with pytest.raises(AttributeError):
            X_kernel = None
            msg = rbf_kernel(X_kernel)
            assert isinstance(msg, AttributeError)


class TestGaussianTaniKernel():
    "Class test for rbf Kernel function"

    def test_output_good_type_and_exists(self):
        """tests that gaussian_tani_kernel output exists with good type

        Returns
        -------
        None
        """
        K = gaussian_tani_kernel(X_kernel)
        # print("XK:",X_kernel)
        # print("K:", K)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        """Tests how gaussian_tani_kernel reacts with bad inputation

        Returns
        -------
        None
        """
        # Test that it handles the case of: Xkernel is a string
        with pytest.raises(AttributeError):
            msg = gaussian_tani_kernel('X_kernel')
            assert isinstance(msg, AttributeError)
        # Test that it handles the case of: Xkernel is None
        with pytest.raises(AttributeError):
            X_kernel = None
            msg = gaussian_tani_kernel(X_kernel)
            assert isinstance(msg, AttributeError)


class TestGaussianComputeGram():
    "Class test for Gaussian.compute_gram function"

    def test_output_good_type_and_exists(self):
        '''Tests that gaussian.compute_gram output exists and returns good type

        Returns
        -------
        None
        '''
        kernel_output = kernel.Gaussian(gamma_output)
        K = kernel_output.compute_gram(X=X_kernel)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        '''Tests how gaussian.compute_gram reacts with bad inputation

        Returns
        -------
        None
        '''
        # Test that it handles the case of: X is a string
        with pytest.raises(TypeError):
            kernel_output = kernel.Gaussian(gamma_output)
            msg = kernel_output.compute_gram('X')
            assert isinstance(msg, TypeError)
        # Test that it handles the case of: X  is None
        with pytest.raises(TypeError):
            X = None
            kernel_output = kernel.Gaussian(gamma_output)
            msg = kernel_output.compute_gram(X)
            assert isinstance(msg, TypeError)


class TestGaussianTaniComputeGram():
    "Class test for Gaussian.compute_gram function"

    def test_output_good_type_and_exists(self):
        """Tests that GaussianTani.compute_gram output exists with good type

        Returns
        -------
        None
        """
        kernel_output = kernel.GaussianTani(gamma_output)
        K = kernel_output.compute_gram(X=X_kernel)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        """Tests that GaussianTani.compute_gram reacts well with bad inputation

        Returns
        -------
        None
        """
        # Test that it handles the case of: X is a string
        with pytest.raises(AttributeError):
            kernel_output = kernel.GaussianTani(gamma_output)
            msg = kernel_output.compute_gram('X')
            assert isinstance(msg, AttributeError)
        # Test that it handles the case of: X  is None
        with pytest.raises(AttributeError):
            X = None
            kernel_output = kernel.GaussianTani(gamma_output)
            msg = kernel_output.compute_gram(X)
            assert isinstance(msg, AttributeError)


class TestLearnableGaussian():
    '''Test class for LearnableGaussian'''

    def test_compute_gram_output_good_type_and_exists(self):
        """Tests that learnablegaussian.compute_gram output exists with good type

        Returns
        -------
        None
        """
        kernel_output = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
        K = kernel_output.compute_gram(X=x_train)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_bad_inputation(self):
        '''Tests that learnablegaussian.compute_gram reacts well with bad inputation

        Returns
        -------
        None
        '''
        # Test that it handles the case of: X is a string
        with pytest.raises(TypeError):
            kernel_output = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
            msg = kernel_output.compute_gram('X')
            assert isinstance(msg, TypeError)
        # Test that it handles the case of: X  is None
        with pytest.raises(TypeError):
            X = None
            kernel_output = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
            msg = kernel_output.compute_gram(X)
            assert isinstance(msg, TypeError)

    def test_clone_kernel_output_same_as_origin(self):
        """Tests that learnablegaussian.clone_kernel works as intended

        Returns
        -------
        None
        """
        origin_kernel = kernel.LearnableGaussian(gamma_input, model_kernel_input, optim_params)
        cloned_kernel = origin_kernel.clone_kernel()
        # assert cloned_kernel.optim_params == kernel_output.optim_params
        # assert cloned_kernel.model == kernel_output.model
        assert type(
            cloned_kernel) == kernel.LearnableGaussian, f"type of 'cloned_kernel' should be the same as 'origin_kernel"


class TestLearnableLinear():
    """class test for LearnableLinear"""

    def test_compute_gram_output_good_type_and_exists(self):
        """Tests that LearnableLinear.compute_gram output exists with good type

        Returns
        -------
        None
        """
        kernel_output = kernel.LearnableLinear(model_kernel_input, optim_params)
        K = kernel_output.compute_gram(X=x_train)
        assert type(K) == torch.Tensor, f"'K' should be a torch.Tensor, but is {type(K)}"
        assert K is not None, f"'K should not be None"
        assert K != "", f"'K' should not be empty"

    def test_clone_kernel_output_same_type_as_origin(self):
        """Tests that learnablelinear.clone_kernel works as intended

        Returns
        -------
        None
        """
        origin_kernel = kernel.LearnableLinear(model_kernel_input, optim_params)
        cloned_kernel = origin_kernel.clone_kernel()
        # assert cloned_kernel.optim_params == kernel_output.optim_params
        # assert cloned_kernel.model == kernel_output.model
        assert type(
            cloned_kernel) == kernel.LearnableLinear, f"type of 'cloned_kernel' should be the same as 'origin_kernel"
