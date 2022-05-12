from os.path import join

import pytest
import torch
from stpredictions.datasets.load_data import load_bibtex_train_from_arff, load_bibtex_test_from_arff

from stpredictions.models.DIOKR import IOKR
from stpredictions.models.DIOKR import cost
from stpredictions.models.DIOKR import estimator
from stpredictions.models.DIOKR import kernel
from stpredictions.models.DIOKR.utils import project_root

dtype = torch.float

# DATA
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

# OTHER
SOLVERS = {
    "sgd": "sgd",
    "adam": "adam",
    # "iris": {'X': iris[0], 'Y': iris[1]},
}
# @pytest.mark.parametrize("key, solver", SOLVERS.items())

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

optim_params = dict(lr=0.01, momentum=0, dampening=0,
                    weight_decay=0, nesterov=False)
kernel_input = kernel.LearnableGaussian(
    gamma_input, model_kernel_input, optim_params)

iokr = IOKR()

diokr_estimator = estimator.DIOKREstimator(kernel_input, kernel_output,
                                           lbda, iokr=iokr, cost=cost_function)


# diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test, n_epochs=20, solver='sgd', batch_size_train=batch_size_train)

# obj = diokr_estimator.objective(x_batch = x_train , y_batch = y_train)

class TestObjective():
    """Test class for 'objective' function"""

    def test_objective_returns_good_type_and_size(self):
        """Test that objective returns good type

        Returns
        -------
        None
        """
        obj = diokr_estimator.objective(x_batch=x_train, y_batch=y_train)
        assert type(obj) == torch.Tensor, f"'obj' should be of type 'torch.Tensor', but is of type {type(obj)}"
        # assert obj.Size() == ,

    def test_bad_inputation(self, ):
        """Tests for raised exceptions (To complete)
        Returns
        -------
        None
        """
        with pytest.raises(TypeError) as exception:
            # Insert a string into the X array
            Xt, yt = x_train, y_train
            Xt[1] = "A string"
            obj = diokr_estimator.objective(x_batch=Xt, y_batch=yt)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        with pytest.raises(TypeError) as exception:
            # Insert a string into the X array
            Xt, yt = x_train, y_train
            yt[1] = "A string"
            obj = diokr_estimator.objective(x_batch=Xt, y_batch=yt)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        # Test that it handles the case of: X is a string
        with pytest.raises(AttributeError):
            Xt, yt = x_train, y_train
            msg = diokr_estimator.objective(x_batch='Xt', y_batch=yt)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"
        # Test that it handles the case of: y is a string
        with pytest.raises(AttributeError):
            Xt, yt = x_train, y_train
            msg = diokr_estimator.objective(x_batch=Xt, y_batch='yt')
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"


class TestTrainKernelInput():
    """Test class for train_kernel_input function"""

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_train_kernel_input_returns_good_type(self, key, solver):
        """Test that train_kernel_input returns good type

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        mse_train = diokr_estimator.train_kernel_input(x_batch=x_train,
                                                       y_batch=y_train,
                                                       solver=solver,
                                                       t0=0)
        assert type(mse_train) == float, f"'mse_train' should be a float, but is {type(mse_train)} instead"
        assert mse_train >= 0, f"'mse_train cannot be negative, and is of {mse_train}"

    def test_solver_bad_inputation(self, ):
        """Tests for raised exceptions (To complete)
        Returns
        -------
        None
        """
        with pytest.raises(ValueError):
            # Insert None in solver
            Xt, yt = x_train, y_train
            solver = None
            msg = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                     y_batch=yt,
                                                     solver=solver,
                                                     t0=0)
            assert isinstance(msg, ValueError)
            assert msg.args[0] == f"'solver' should be 'sgd' or 'adam',but not {solver}"
        # Test that it handles the case of: solver is not a string
        with pytest.raises(ValueError):
            Xt, yt = x_train, y_train
            solver = 0
            msg = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                     y_batch=yt,
                                                     solver=solver,
                                                     t0=0)
            assert isinstance(msg, ValueError)
            assert msg.args[0] == f"'solver' should be 'sgd' or 'adam',but not {solver}"

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_t0_bad_inputation(self, key, solver):
        """Test train_kernel_input when t0 is a bad inputation

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        with pytest.raises(TypeError):
            # Insert None in t0
            Xt, yt = x_train, y_train
            t0 = None
            msg = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                     y_batch=yt,
                                                     solver=solver,
                                                     t0=t0)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f"'t0' should be an int, not a {type(t0)}"
        # Test that it handles the case of: tO is a string
        with pytest.raises(TypeError):
            Xt, yt = x_train, y_train
            t0 = 't0'
            msg = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                     y_batch=yt,
                                                     solver=solver,
                                                     t0=t0)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f"'t0' should be an int, not a {type(t0)}"

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_XY_batch_bad_inputation(self, key, solver):
        """Test train_kernel_input when X/Y batch has bad inputation

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        with pytest.raises(TypeError) as exception:
            # Insert a string into the X array
            Xt, yt = x_train, y_train
            Xt[1] = "A string"
            mse_train = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                           y_batch=yt,
                                                           solver=solver,
                                                           t0=0)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        with pytest.raises(TypeError) as exception:
            # Insert a string into the y array
            Xt, yt = x_train, y_train
            yt[1] = "A string"
            mse_train = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                           y_batch=yt,
                                                           solver=solver,
                                                           t0=0)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        # Test that it handles the case of: X is a string
        with pytest.raises(AttributeError):
            Xt, yt = x_train, y_train
            msg = diokr_estimator.train_kernel_input(x_batch='Xt',
                                                     y_batch=yt,
                                                     solver=solver,
                                                     t0=0)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"
        # Test that it handles the case of: y is a string
        with pytest.raises(AttributeError):
            Xt, yt = x_train, y_train
            msg = diokr_estimator.train_kernel_input(x_batch=Xt,
                                                     y_batch='yt',
                                                     solver=solver,
                                                     t0=0)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"


class TestFitKernelInput():

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_verbose(self, capfd, key, solver):
        """Test if fit function actually prints something
         Parameters
         ----------
         capfd: fixture
         Allows access to stdout/stderr output created
         during test execution.
         Returns
         -------
         None
         """

        """Test if fit function actually prints something"""
        scores = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                  n_epochs=5, solver=solver, batch_size_train=batch_size_train)
        out, err = capfd.readouterr()
        assert err == "", f'{err}: need to be fixed'
        assert out != "", f'Each Epoch run and MSE should have been printed'

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_n_epochs_bad_inputation(self, key, solver):
        """Tests for raised exceptions (To complete)
       Returns
       -------
       None
       """
        with pytest.raises(TypeError):
            # Insert None in epochs
            n_epochs = None
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=n_epochs, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f" 'n_epochs' should be an int, not a {type(n_epochs)}"
        # Test that it handles the case of: n_epoch is not a int
        with pytest.raises(TypeError):
            Xt, yt = x_train, y_train
            n_epochs = '5'
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=n_epochs, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f"'n_epochs' should be an int, not a {type(n_epochs)}"

    def test_solver_bad_inputation(self, ):
        """Test fit_kernel_inputation when solver has bad inputation

        Returns
        -------
        None
        """
        with pytest.raises(ValueError):
            # Insert None in solver
            solver = None
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, ValueError)
            assert msg.args[0] == f"'solver' should be 'sgd' or 'adam',but not {solver}"
        # Test that it handles the case of: solver is not a string
        with pytest.raises(ValueError):
            Xt, yt = x_train, y_train
            solver = 0
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, ValueError)
            assert msg.args[0] == f"'solver' should be 'sgd' or 'adam',but not {solver}"

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_batch_size_bad_inputation(self, key, solver):
        """Test fit_kernel_inputation when batch_size has bad inputation

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        with pytest.raises(TypeError):
            # Insert None in batchsize
            Xt, yt = x_train, y_train
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=None)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f"'batch_size_train' should be an int, not a {type(batch_size_train)}"
        # Test that it handles the case of: batchsize is a string
        with pytest.raises(TypeError):
            Xt, yt = x_train, y_train
            bst = 'batch_size_train'
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=bst)
            assert isinstance(msg, TypeError)
            assert msg.args[0] == f"'batch_size_train' should be an int, not a {type(bst)}"

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_XY_bad_inputation(self, key, solver):
        """Test fit_kernel_input when XY has bad inputation

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        with pytest.raises(TypeError) as exception:
            # Insert a string into the X array
            Xt, yt = x_train, y_train
            Xt[1] = "A string"
            diokr_estimator.fit_kernel_input(Xt, yt, x_test, y_test,
                                             n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        with pytest.raises(TypeError) as exception:
            # Insert a string into the y array
            Xt, yt = x_train, y_train
            yt[1] = "A string"
            diokr_estimator.fit_kernel_input(Xt, yt, x_test, y_test,
                                             n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        with pytest.raises(TypeError) as exception:
            # Insert a string into the Xtest array
            Xt, yt = x_test, y_test
            Xt[1] = "A string"
            diokr_estimator.fit_kernel_input(x_train, y_train, Xt, yt,
                                             n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        with pytest.raises(TypeError) as exception:
            # Insert a string into the ytest array
            Xt, yt = x_test, y_test
            yt[1] = "A string"
            diokr_estimator.fit_kernel_input(x_train, y_train, Xt, yt,
                                             n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert "can't assign a str to a torch.FloatTensor" in str(exception.value)
        # Test that it handles the case of: X is a string
        with pytest.raises(AttributeError):
            msg = diokr_estimator.fit_kernel_input('x_train', y_train, x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"
        # Test that it handles the case of: y is a string
        with pytest.raises(AttributeError):
            msg = diokr_estimator.fit_kernel_input(x_train, 'y_train', x_test, y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"
        # Test that it handles the case of: Xtest is a string
        with pytest.raises(AttributeError):
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, 'x_test', y_test,
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"
        # Test that it handles the case of: ytest is a string
        with pytest.raises(AttributeError):
            msg = diokr_estimator.fit_kernel_input(x_train, y_train, x_test, 'y_test',
                                                   n_epochs=5, solver=solver, batch_size_train=batch_size_train)
            assert isinstance(msg, AttributeError)
            assert msg.args[0] == "'str' object has no attribute 'clone'"


class TestPredict():
    """Test class for predict function"""

    @pytest.mark.parametrize("key, solver", SOLVERS.items())
    def test_predict_returns_good_type_and_shape(self, key, solver):
        """Test that predict function return good type and shape

        Parameters
        ----------
        key : fixture/str
        key of the dict solver
        solver : fixture/str
        value of the dict solver

        Returns
        -------
        None
        """
        diokr_estimator.fit_kernel_input(x_train, y_train, x_test, y_test,
                                         n_epochs=10, solver=solver, batch_size_train=batch_size_train)
        Y_pred_test = diokr_estimator.predict(x_test=x_test)
        assert type(Y_pred_test) == torch.Tensor, f"'Y_pred' should be a torch.Tensor, but is {type(Y_pred_test)}"
        assert Y_pred_test.size() == (x_test.shape[0], y_test.shape[1]), f"Wrong shape for 'Y_pred_test':" \
                                                                         f"should be: ({x_test.shape[0]}, {y_test.shape[1]})" \
                                                                         f"Is: {Y_pred_test.size()}"
