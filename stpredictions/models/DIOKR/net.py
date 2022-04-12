import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self, dim_inputs, dim_outputs):
        super(Net1, self).__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)
        self.dim_inputs = dim_inputs

    def forward(self, x):
        x = x.view(-1, self.dim_inputs)  # reshape input to batch x num_inputs
        x = self.linear(x)
        x = F.relu(x)
        return x

    def get_layers(self):
        dict_layers = {}
        dict_layers['linear.weight'] = self.linear.weight
        dict_layers['linear.bias'] = self.linear.bias
        return dict_layers


class Net2(nn.Module):

    def __init__(self, dim_inputs, dim_outputs):
        super(Net2, self).__init__()
        self.linear1 = nn.Linear(dim_inputs, 1000)
        self.dim_inputs = dim_inputs
        self.linear2 = nn.Linear(1000, dim_outputs)

    def forward(self, x):
        x = x.view(-1, self.dim_inputs)  # reshape input to batch x num_inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = x.view(-1, 1000)  # reshape input to batch x num_inputs
        x = self.linear2(x)
        x = F.relu(x)
        return x

    def get_layers(self):
        dict_layers = {}
        dict_layers['linear.weight'] = self.linear1.weight
        dict_layers['linear.bias'] = self.linear1.bias
        dict_layers['linear.weight'] = self.linear2.weight
        dict_layers['linear.bias'] = self.linear2.bias
        return dict_layers


class Net3(nn.Module):

    def __init__(self, dim_inputs, dim_outputs):
        super(Net3, self).__init__()
        self.linear1 = nn.Linear(dim_inputs, 1024)
        self.dim_inputs = dim_inputs
        self.linear2 = nn.Linear(1024, 600)
        self.linear3 = nn.Linear(600, dim_outputs)

    def forward(self, x):
        x = x.view(-1, self.dim_inputs)  # reshape input to batch x num_inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = x.view(-1, 1024)  # reshape input to batch x num_inputs
        x = self.linear2(x)
        x = F.relu(x)
        x = x.view(-1, 600)  # reshape input to batch x num_inputs
        x = self.linear3(x)
        x = F.relu(x)
        return x

    def get_layers(self):
        dict_layers = {}
        dict_layers['linear.weight'] = self.linear1.weight
        dict_layers['linear.bias'] = self.linear1.bias
        dict_layers['linear.weight'] = self.linear2.weight
        dict_layers['linear.bias'] = self.linear2.bias
        dict_layers['linear.weight'] = self.linear3.weight
        dict_layers['linear.bias'] = self.linear3.bias
        return dict_layers
