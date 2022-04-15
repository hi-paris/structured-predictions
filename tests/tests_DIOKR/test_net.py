# from stpredictions.models.DIOKR import cost
# from stpredictions.models.DIOKR import kernel
# from stpredictions.models.DIOKR import IOKR
# from stpredictions.models.DIOKR import estimator
from os.path import join

import torch
from stpredictions.datasets.load_data import load_bibtex_train_from_arff
from stpredictions.models.DIOKR.utils import project_root


from stpredictions.models.DIOKR import net
from stpredictions.models.DIOKR.utils import project_root


x_train, y_train = load_bibtex_train_from_arff()
x_train, y_train = x_train.todense(), y_train.todense()

x_train, y_train = x_train[:2000], y_train[:2000]

x_train = torch.from_numpy(x_train).float()


# dim_inputs = x_train.shape[1]
# dim_outputs = int(dim_inputs/2)
# x = net.Net1(dim_inputs, dim_outputs)
# x = x.get_layers()
# print(x)
# print(type(x))

class TestNet1():
    """Test class for Net1 function"""

    def test_net_instance_good_shape(self, capfd):
        """Test that a Net1 instance has good shape

        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self, ):
        """Tests that a net1 instance has good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        """Test that net.get_layers function returns good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net1(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"


class TestNet2():

    def test_net_instance_good_shape(self, capfd):
        """Test that a Net2 instance has good shape

        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self):
        """Tests that a net1 instance has good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        """Test that net.get_layers function returns good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net2(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"


class TestNet3():

    def test_net_instance_good_shape(self, capfd):
        """Test that a Net3 instance has good shape

        Parameters
        ----------
        capfd: fixture
        Allows access to stdout/stderr output created
        during test execution.

        Returns
        -------
        None
        """

        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        print(x)
        out, err = capfd.readouterr()
        assert out != f"Net1((linear): Linear(in_features={dim_inputs}, out_features={dim_outputs}, bias=True))"

    def test_net_forward_returns_good_type(self, ):
        """Tests that a net1 instance has good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        x = x.forward(x_train)
        assert type(x) == torch.Tensor, f"'x' should be a torch.Tensor, but is {type(x)}"

    def test_net_get_layers_returns_good_type(self, ):
        """Test that net.get_layers function returns good type

        Returns
        -------
        None
        """
        dim_inputs = x_train.shape[1]
        dim_outputs = int(dim_inputs / 2)
        x = net.Net3(dim_inputs, dim_outputs)
        x = x.get_layers()
        assert type(x) == dict, f"'x' should be a dict, but is {type(x)}"
