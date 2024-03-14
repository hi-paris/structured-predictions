import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock,ResNet


class Net1(nn.Module):
    
    def __init__(self, dim_inputs, dim_outputs):
        super(Net1, self).__init__()
        self.linear = nn.Linear(dim_inputs, dim_outputs)
        self.dim_inputs = dim_inputs
        
    def forward(self, x):
        x = x.view(-1, self.dim_inputs) # reshape input to batch x num_inputs
        x = self.linear(x)
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
        x = x.view(-1, self.dim_inputs) # reshape input to batch x num_inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = x.view(-1, 1000) # reshape input to batch x num_inputs
        x = self.linear2(x)
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
        x = x.view(-1, self.dim_inputs) # reshape input to batch x num_inputs
        x = self.linear1(x)
        x = F.relu(x)
        x = x.view(-1, 1024) # reshape input to batch x num_inputs
        x = self.linear2(x)
        x = F.relu(x)
        x = x.view(-1, 600) # reshape input to batch x num_inputs
        x = self.linear3(x)
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
    

class ConvNet(nn.Module):
    def __init__(self, input_shape=(1, 64, 64), output_dim=5000):
        super(ConvNet, self).__init__()
        self.input_c, self.input_w, self.input_h = input_shape
        self.conv1 = nn.Conv2d(self.input_c, 16, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 128, 5, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        input = torch.zeros((64, self.input_c, self.input_w, self.input_h))
        output_shape = self.pool2(self.conv2(self.pool1(self.conv1(input)))).shape
        print(output_shape)
        self.c, self.w, self.h = output_shape[1], output_shape[2], output_shape[3]
        self.fc1 = nn.Linear(self.h * self.w * self.c, 4000)
        self.fc2 = nn.Linear(4000, output_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.h * self.w * self.c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class WHDVNNet(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), output_dim=5000):
        super(WHDVNNet, self).__init__()
        self.input_c, self.input_w, self.input_h = input_shape
        self.conv1 = nn.Conv2d(self.input_c, 64, 5, stride=2, padding=2)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        #self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        input = torch.zeros((64, self.input_c, self.input_w, self.input_h))
        output_shape = self.conv3(self.conv2(self.conv1(input))).shape
        print(output_shape)
        self.c, self.w, self.h = output_shape[1], output_shape[2], output_shape[3]        
        self.fc1 = nn.Linear(self.h * self.w * self.c, 3000)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(3000, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.h * self.w * self.c)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class SmallCNN(nn.Module):

    def __init__(self, model_dim=64, input_shape=(3, 32, 32), output_dim=5000):
        super().__init__()

        self.params = {'model_dim':model_dim,
                       'input_shape':input_shape}

        self.cnn = nn.Sequential(nn.Conv2d(self.params['input_shape'][0], 32, 3, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(32, 64, 3, padding=1, stride=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(64, 128, 3, padding=1, stride=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, model_dim, 1),
                                 )
        input = torch.zeros((64,
                             self.params['input_shape'][0],
                             self.params['input_shape'][1],
                             self.params['input_shape'][2]))
        output_shape = self.cnn(input).shape
        print(output_shape)
        self.c, self.w, self.h = output_shape[1], output_shape[2], output_shape[3]
        
        self.fc1 = nn.Linear(self.h * self.w * self.c, 4500)
        self.fc2 = nn.Linear(4500, output_dim)


    def forward(self, x):

        x = self.cnn(x)
        x = x.view(-1, self.h * self.w * self.c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class MediumCNN(nn.Module):

    def __init__(self, model_dim=64, input_shape=(3, 32, 32), output_dim=5000):
        super().__init__()

        self.params = {'model_dim':model_dim,
                       'input_shape':input_shape}

        self.cnn = nn.Sequential(nn.Conv2d(self.params['input_shape'][0], 32, 3,padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(32, 64, 3,padding=1,stride=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, 3,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(64, 128, 3,padding=1,stride=1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 128, 3),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, model_dim, 1),
                                 )
        
        input = torch.zeros((64,
                             self.params['input_shape'][0],
                             self.params['input_shape'][1],
                             self.params['input_shape'][2]))
        output_shape = self.cnn(input).shape
        print(output_shape)
        self.c, self.w, self.h = output_shape[1], output_shape[2], output_shape[3]
        
        self.fc1 = nn.Linear(self.h * self.w * self.c, 4500)
        self.fc2 = nn.Linear(4500, output_dim)


    def forward(self, x):

        x = self.cnn(x)
        x = x.view(-1, self.h * self.w * self.c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    




def troncated_resnet(number_of_blocks_to_remove=2):
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    model = nn.Sequential(*(list(resnet18.children())[:3] + list(resnet18.children())[4:-2 - number_of_blocks_to_remove]))  # Remove maxpool + 3 and 4th blocks + final linear layers
    return model
    
    
class MyResNet(nn.Module):

    def __init__(self, model_dim=64, input_shape=(3, 32, 32), output_dim=5000):
        super().__init__()

        self.params = {'model_dim':model_dim,
                       'input_shape':input_shape}

        self.cnn = nn.Sequential(troncated_resnet(),
                                 nn.ReLU(),
                                 nn.Conv2d(128, model_dim, 1)) 

        input = torch.zeros((64, input_shape[0], input_shape[1], input_shape[2]))
        output_shape = self.cnn(input).shape
        print(output_shape)
        self.c, self.w, self.h = output_shape[1], output_shape[2], output_shape[3]
        
        self.fc1 = nn.Linear(self.h * self.w * self.c, 4500)
        self.fc2 = nn.Linear(4500, output_dim)


    def forward(self, x):

        x = self.cnn(x)
        x = x.view(-1, self.h * self.w * self.c)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x