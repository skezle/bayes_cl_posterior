import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## thanks: https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, seed, in_dim, out_dim, hid_dim, cl3=False):
        super(Net, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.task_idx = 1
        self.cl3 = cl3

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output